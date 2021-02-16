import numpy as np
import numbers
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

def pad_load(rho,load,mesh):
    """Sets all elements containing a node subject to a load to solid"""
    batch_size,_,nely,nelx = rho.shape
    pad_mat = torch.ones(rho.shape).to(rho.device)
    for i in range(batch_size):
        load_dof = np.nonzero(load[i].cpu().numpy())[0]
        node = np.unique(mesh.dof2nodeid(load_dof))
        node_ele_nbrs = np.nonzero(mesh.IX==node)[0]
        elx = torch.tensor(node_ele_nbrs//nely,dtype=torch.long)
        ely = torch.tensor(node_ele_nbrs%nely,dtype=torch.long)
        pad_mat[i,0,ely,elx] = 1/rho[i,0,ely,elx]
    rho = rho*pad_mat
    return rho

class ConvBatchReLU(nn.Module):
    """Convience class for performing Conv2d + BatchNorm2d + ReLU"""

    def __init__(self, n_in, n_out, kernel_size, stride=1, dilation=1, groups=1, padding='same'):
        super(ConvBatchReLU, self).__init__()
        if padding=='same':
            self.npad = (kernel_size+(kernel_size-1)*(dilation-1)-1)//2
        elif padding=='none':
            self.npad = 0
        else:
            raise ValueError("Padding method not implemented")

        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, padding=self.npad),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        out = self.conv(x)
        return out

class ResNetBlock(nn.Module):
    """ Class for creating a residual network block, where the network
    predicts the residual between the input x and output y:
    y = f(x) + x -> f(x) = y - x
    """

    def __init__(self, n_channels,kernel_size,dilation=1):
        super(ResNetBlock, self).__init__()
        self.npad = (kernel_size+(kernel_size-1)*(dilation-1)-1)//2
        self.residual = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, dilation=dilation, stride=1,padding=self.npad),
            nn.BatchNorm2d(n_channels),
            nn.LeakyReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, dilation=dilation, stride=1,padding=self.npad),
        )
    def forward(self, x):

        res = self.residual(x)
        out = F.leaky_relu(x+res)
        return out

class SE_Block(nn.Module):
    """Squeeze-and-excite block used to perform dynamic channel-wise feature recalibration,
    inspired by Nie et al. 2020 'TopologyGAN' """
    def __init__(self,shape_in):
        super(SE_Block, self).__init__()
        B,C,H,W = shape_in
        self.reduction_factor = 2

        self.GlobalAvgPool = nn.AvgPool2d(kernel_size=(H,W))
        self.fc1 = nn.Sequential(
            nn.Linear(C,C//self.reduction_factor),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(C//self.reduction_factor,C),
            nn.Sigmoid()
        )

    def forward(self, x):
        B,C,H,W = x.shape
        x_avg = self.GlobalAvgPool(x)
        x_avg = x_avg.view((B,C))
        xs = self.fc1(x_avg)
        xe = self.fc2(xs)
        Ex = xe.view((B,C,1,1))
        out = Ex*x
        return out

class SE_ResNetBlock(nn.Module):
    """Combines the ResNetBlock and SE_block into a single module"""
    def __init__(self, n_channels,shape_in,kernel_size,dilation=1):
        super(SE_ResNetBlock, self).__init__()
        self.residual = nn.Sequential(
            ConvBatchReLU(n_channels, n_channels, kernel_size=kernel_size, dilation=dilation, stride=1),
            ConvBatchReLU(n_channels, n_channels, kernel_size=kernel_size, dilation=dilation, stride=1),
        )
        self.se = SE_Block(shape_in)

    def forward(self, x):
        u = self.residual(x)
        v = self.se(u)
        out = F.leaky_relu(x+v)
        return out

class InputLayer(nn.Module):
    """Input layer used to process von mises stresses, volume field and
    the principal streamline image separately"""
    def __init__(self,include_psi=1,num_input_ch=16):
        super(InputLayer,self).__init__()
        self.include_psi = include_psi
        n_ch_sigma_vol = num_input_ch
        n_ch_psi = num_input_ch
        self.input_size = n_ch_sigma_vol + include_psi*n_ch_psi
        self.sigma_vol_in = nn.Sequential(
           ConvBatchReLU(2,n_ch_sigma_vol,kernel_size=3),
           ConvBatchReLU(n_ch_sigma_vol,n_ch_sigma_vol,kernel_size=3),
        )
        self.psi_in = nn.Sequential(
            ConvBatchReLU(1,n_ch_psi,kernel_size=3),
            ConvBatchReLU(n_ch_psi,n_ch_psi,kernel_size=3),
            ConvBatchReLU(n_ch_psi,n_ch_psi,kernel_size=3,stride=2),
            ConvBatchReLU(n_ch_psi,n_ch_psi,kernel_size=3),
            ConvBatchReLU(n_ch_psi,n_ch_psi,kernel_size=3,stride=2),
            ConvBatchReLU(n_ch_psi,n_ch_psi,kernel_size=3),
        )
    def forward(self,sigma,psi,vol_field):
        if self.include_psi==1:
            x0 = self.sigma_vol_in(torch.cat([sigma,vol_field],axis=1))
            x1 = self.psi_in(psi)
            out = torch.cat([x0,x1],axis=1)
        else:
            out = self.sigma_vol_in(torch.cat([sigma,vol_field],axis=1))
        return out

class TopOptNet(nn.Module):
    """U-SE-ResNet8 model used for training on the MBC dataset"""
    def __init__(self,shape_in,skip_conn=1,num_res_blocks=8):
        super(TopOptNet,self).__init__()
        # specify network parameters
        self.skip_conn = skip_conn
        B,H,W = shape_in # batch size, height and width
        Hb = int(np.ceil(H/(2**3))) # height in bottleneck layer
        Wb = int(np.ceil(W/(2**3))) # width in bottleneck layer
        # number of channels in each layer
        n_ch_input = 16
        n_ch_enc0 = 16
        n_ch_enc1 = 32
        n_ch_enc2 = 64
        n_ch_bneck = 128
        n_ch_dec0 = 64
        n_ch_dec1 = 32
        n_ch_dec2 = 16
        # layer definitions
        self.input_layer = InputLayer(include_psi=1,num_input_ch=n_ch_input)
        # encoder
        self.enc_conv0 = nn.Sequential(
            ConvBatchReLU(self.input_layer.input_size ,n_ch_enc0,kernel_size=3),
            ConvBatchReLU(n_ch_enc0,n_ch_enc0,kernel_size=3),
        )
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc_conv1 = nn.Sequential(
            ConvBatchReLU(n_ch_enc0,n_ch_enc1,kernel_size=3),
            ConvBatchReLU(n_ch_enc1,n_ch_enc1,kernel_size=3),
            ConvBatchReLU(n_ch_enc1,n_ch_enc1,kernel_size=3),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Sequential(
            ConvBatchReLU(n_ch_enc1,n_ch_enc2,kernel_size=3),
            ConvBatchReLU(n_ch_enc2,n_ch_enc2,kernel_size=3),
            ConvBatchReLU(n_ch_enc2,n_ch_enc2,kernel_size=3),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # bottleneck
        res_net_layers = []
        res_input_layer = ConvBatchReLU(n_ch_enc2,n_ch_bneck,kernel_size=3)
        res_net_layers.append(res_input_layer)
        for i in range(num_res_blocks):
            res_net_layers.append(SE_ResNetBlock(n_ch_bneck,(B,n_ch_bneck,Hb,Wb),kernel_size=3))
        self.bottleneck = nn.Sequential(*res_net_layers)
        # decoder
        self.upsample0 = nn.Upsample(size=(15,30),mode="bilinear")
        self.dec_conv0 = nn.Sequential(
            ConvBatchReLU(n_ch_bneck+n_ch_enc2*skip_conn,n_ch_dec2,kernel_size=3),
            ConvBatchReLU(n_ch_dec2,n_ch_dec2,kernel_size=3),
            ConvBatchReLU(n_ch_dec2,n_ch_dec1,kernel_size=3),
        )
        self.upsample1 = nn.Upsample(scale_factor=2,mode="bilinear")
        self.dec_conv1 = nn.Sequential(
            ConvBatchReLU(n_ch_dec1+n_ch_enc1*skip_conn,n_ch_dec1,kernel_size=3),
            ConvBatchReLU(n_ch_dec1,n_ch_dec1,kernel_size=3),
            ConvBatchReLU(n_ch_dec1,n_ch_dec2,kernel_size=3),
        )
        self.upsample2 = nn.Upsample(scale_factor=2,mode="bilinear")
        self.dec_conv2 = nn.Sequential(
            ConvBatchReLU(n_ch_dec2+n_ch_enc0*skip_conn,n_ch_dec2,kernel_size=3),
            ConvBatchReLU(n_ch_dec2,n_ch_dec2,kernel_size=3),
            ConvBatchReLU(n_ch_dec2,n_ch_dec2,kernel_size=3),
        )
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_ch_dec2, 1, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self,sigma,psi,vol_field,load,mesh):
        x = self.input_layer(sigma,psi,vol_field)
        # encoder
        ec0 = self.enc_conv0(x)
        ep0 = self.pool0(ec0)
        ec1 = self.enc_conv1(ep0)
        ep1 = self.pool1(ec1)
        ec2 = self.enc_conv2(ep1)
        ep2 = self.pool2(ec2)
        # bottleneck
        b = self.bottleneck(ep2)
        # decoder
        du0 = self.upsample0(b)
        if self.skip_conn==1:
            dc0 = self.dec_conv0(torch.cat([du0, ec2], 1))
        else:
            dc0 = self.dec_conv0(du0)
        du1 = self.upsample1(dc0)
        if self.skip_conn==1:
            dc1 = self.dec_conv1(torch.cat([du1, ec1], 1))
        else:
            dc1 = self.dec_conv1(du1)
        du2 = self.upsample2(dc1)
        if self.skip_conn==1:
            dc2 = self.dec_conv2(torch.cat([du2, ec0], 1))
        else:
            dc2 = self.dec_conv2(du2)
        rho = self.output_layer(dc2)
        rho = pad_load(rho,load,mesh)
        return rho

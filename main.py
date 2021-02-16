import argparse
import os
import sys
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL.Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# local libraries
from FEA import FEmesh, LinearElasticity
from models import TopOptNet
from spalg import ComplianceLoss, batch_solve
from utils import count_model_parameters, tensorboard_plot

class TrainData(torch.utils.data.Dataset):
    """Class for storing the training dataset
    Attributes:
    data_path(str): location of the training datafile
    transform(torchvision.transforms object): transformations to be applied to the training data
    train_data(numpy object): storage of training data in memory
    max_fixed_dofs: maximum number of fixed dofs for a single sample in the dataset
    """
    def __init__(self,transform,data_path):
        self.data_path = data_path
        self.transform = transform
        self.train_data = np.load(os.path.join(data_path,"train.npy"),allow_pickle=True)
        self.max_fixed_dofs = max([len(self.train_data[i][3]) for i in range(len(self.train_data))])

    def __len__(self):
        "Mandatory function which returns the length of the dataset"
        return len(self.train_data)

    def __getitem__(self, idx):
        "Mandatory function which returns a single sample from the dataset"
        sigma_vm,psi_strm,vol_field,fixed_dofs,load,solid_comp = self.train_data[idx]
        sigma_vm = self.ToFloatTensor(sigma_vm)
        psi_strm = self.ToFloatTensor(psi_strm)
        vol_field = self.ToFloatTensor(vol_field)
        # pad the fixed dofs array with -1 (each sample may contain different amount of fixed_dofs)
        padded_fixed_dofs = -np.ones(self.max_fixed_dofs,dtype=int)
        padded_fixed_dofs[:len(fixed_dofs)] = fixed_dofs
        return sigma_vm,psi_strm,vol_field,padded_fixed_dofs,load,solid_comp

    def ToFloatTensor(self,x):
        """Converts a numpy array to a torch float tensor and applies the defined transformations"""
        return self.transform(x.astype(np.float32))

class TestData(torch.utils.data.Dataset):
    """Same as TrainData class, but for the test set instead. Requires an additional file 'topopt.npy'
    which contains the baseline designs"""
    def __init__(self,transform,data_path):
        self.data_path = data_path
        self.transform = transform
        self.test_data = np.load(os.path.join(data_path,"test.npy"),allow_pickle=True)
        self.topopt_data = np.load(os.path.join(data_path,"topopt.npy"),allow_pickle=True)
        self.max_fixed_dofs = max([len(self.test_data[i][3]) for i in range(len(self.test_data))])

    def __len__(self):
        "Mandatory function which returns the length of the dataset"
        return len(self.test_data)

    def __getitem__(self, idx):
        "Mandatory function which returns a single sample from the dataset"
        sigma_vm,psi_strm,vol_field,fixed_dofs,load,solid_comp = self.test_data[idx]
        rho_topopt = self.topopt_data[idx]
        sigma_vm = self.ToFloatTensor(sigma_vm)
        psi_strm = self.ToFloatTensor(psi_strm)
        vol_field = self.ToFloatTensor(vol_field)
        # pad the fixed dofs array with -1 (each sample may contain different amount of fixed_dofs)
        padded_fixed_dofs = -np.ones(self.max_fixed_dofs,dtype=int)
        padded_fixed_dofs[:len(fixed_dofs)] = fixed_dofs
        return sigma_vm,psi_strm,vol_field,padded_fixed_dofs,load,solid_comp,rho_topopt

    def ToFloatTensor(self,x):
        """Converts a numpy array to a torch float tensor and applies the defined transformations"""
        return self.transform(x.astype(np.float32))

def eval_testset_compliance(test_loader,loss_fn,model,mesh,fea,dens_penal):
    """Evaluates the model on the testset and returns the average compliance"""
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    batch_size = test_loader.batch_size
    dset_size = batch_size*len(test_loader)
    comp_arr = torch.zeros(dset_size).to(device)
    for it,input_data in enumerate(test_loader):
        with torch.no_grad():
            sigma_vm,psi_strm,vol_field,fixed_dofs,load,solid_comp,rho_topopt = input_data
            sigma_vm = sigma_vm.to(device)
            psi_strm = psi_strm.to(device)
            vol_field = vol_field.to(device)
            rho = model(sigma_vm,psi_strm,vol_field,load,mesh)
            # solve FE for each entry in batch
            U, _ = batch_solve(fea,rho,fixed_dofs,load,dens_penal,von_mises=False)
            # compute loss
            solid_comp = solid_comp.to(device)
            vol_field = vol_field.to(device)
            comp_loss, comp, vol_constr = loss_fn(rho,U,vol_field,solid_comp,dens_penal,lambda_vol=1e1)
            comp_arr[it*batch_size:(it+1)*batch_size] = comp

    return torch.mean(comp_arr).item()

def write_config_file(filepath,args,model):
    """Writes a configuration file to store the model and training parameters"""
    timestamp_str = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    config_filename = "config_"+timestamp_str+".txt"
    f = open(os.path.join(filepath,config_filename), "w")
    f.write("Data: "+str(args.DATA_PATH)+"\n")
    f.write("Pretrained: "+str(args.pretrained)+"\n")
    f.write("Batch size: "+str(args.batch_size)+"\n")
    f.write("Dens penal: "+str(args.dens_penal)+"\n")
    f.write("Vol penal: "+str(args.vol_penal)+"\n")
    f.write("Learning rate: "+str(args.lr)+"\n")
    f.write("Clip value: "+str(args.clip)+"\n")
    f.write("Model: "+model._get_name()+"\n")
    f.close()

def run(args):
    """Main function used for training the model"""
    # Define model name and save path
    MODEL_PATH = os.path.join(args.SAVE_PATH,args.model_name)
    if os.path.exists(MODEL_PATH) is False:
        os.makedirs(MODEL_PATH)
        os.makedirs(os.path.join(MODEL_PATH,"weights"))

    # Get device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # Create dataloaders
    batch_size = args.batch_size
    train_dset = TrainData(transform=transforms.ToTensor(),data_path=args.DATA_PATH)
    train_loader = DataLoader(train_dset, batch_size=batch_size,shuffle=True,
                          num_workers=n_gpu*4,pin_memory=(torch.cuda.is_available()),drop_last=True)
    test_dset = TestData(transform=transforms.ToTensor(),data_path=args.DATA_PATH)
    test_loader = DataLoader(test_dset, batch_size=batch_size,shuffle=False,
                      num_workers=n_gpu*4,pin_memory=(torch.cuda.is_available()),drop_last=False)

    # Create finite element mesh
    input_data = next(iter(train_loader))
    input_dim = input_data[0].shape
    nelx = input_dim[-1]
    nely = input_dim[-2]
    mesh = FEmesh(nelx,nely)
    # Initialize linear elasticity class
    fea = LinearElasticity(mesh)

    # Initialize model and optimizer
    model = TopOptNet(shape_in=(batch_size,nely,nelx))
    print("Running model on",n_gpu,"GPUs")
    if len(args.pretrained)>0:
        model.load_state_dict(torch.load(args.pretrained),strict=True)
        print("Using pretrained model")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=0.01)
    print("Model parameters:",count_model_parameters(model))

    # Write file with configurations for the training
    write_config_file(MODEL_PATH,args,model)

    # Initialize loss function and continuation schemes
    comp_loss_fn = ComplianceLoss(mesh,fea)
    n_epochs = args.epochs
    vol_penal = args.vol_penal
    max_vol_penal = 1e2
    dens_penal = args.dens_penal
    summ_writer = SummaryWriter(log_dir=MODEL_PATH)
    # training loop
    it=0
    for epoch in range(n_epochs):
        for batch_nr,input_data in enumerate(tqdm(train_loader)):
            model.zero_grad()
            # parse input through model
            sigma_vm,psi_strm,vol_field,fixed_dofs,load,solid_comp = input_data
            sigma_vm = sigma_vm.to(device)
            psi_strm = psi_strm.to(device)
            vol_field = vol_field.to(device)
            rho = model(sigma_vm,psi_strm,vol_field,load,mesh)
            # solve FE for each entry in batch
            with torch.no_grad(): # do not include # IDEA: n computational graph
                U, _ = batch_solve(fea,rho,fixed_dofs,load,dens_penal,von_mises=False)
            # compute loss
            solid_comp = solid_comp.to(device)
            vol_field = vol_field.to(device)
            comp_loss, comp, vol_constr = comp_loss_fn(rho,U,vol_field,solid_comp,dens_penal,lambda_vol=vol_penal)
            # back propagate
            batch_loss = torch.mean(comp_loss)
            batch_comp = torch.mean(comp/solid_comp)
            batch_vol = torch.mean(vol_constr)
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            it+=1
            # store variables in tensorboard
            summ_writer.add_scalar('Loss', batch_loss.item(), it)
            summ_writer.add_scalar('Compliance',batch_comp.item(), it)
            summ_writer.add_scalar('Vol. constr.',batch_vol.item(), it)
            # store image of designs every 50th epoch
            if batch_nr%50==0:
                designs_image = PIL.Image.open(tensorboard_plot(mesh,rho,vol_field,fixed_dofs,load))
                designs_image = transforms.ToTensor()(designs_image)
                summ_writer.add_image('Design samples', designs_image, it)

        # update volume constraint
        if (epoch>=10):
            if vol_penal<max_vol_penal:
                vol_penal*=1.05
                print("Increasing volume penalization to",vol_penal)
            else:
                vol_penal = max_vol_penal

        #calculate test set compliance
        #test_comp = eval_testset_compliance(test_loader,comp_loss_fn,model,mesh,fea,dens_penal)
        #print("Test set compliance:",test_comp)
        #summ_writer.add_scalar('Test comp.', test_comp, epoch)

        print("Saving model at "+MODEL_PATH)
        torch.save(model.state_dict(), os.path.join(MODEL_PATH,"weights","epoch"+str(epoch)+".pth"))
    # close the tensorboard writer
    summ_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('SAVE_PATH')
    parser.add_argument('model_name')
    parser.add_argument('DATA_PATH')
    parser.add_argument('--pretrained',default="")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--clip', default=1, type=float)
    parser.add_argument('--dens_penal', default=2, type=float)
    parser.add_argument('--vol_penal', default=1e1, type=float)
    args = parser.parse_args()

    run(args)

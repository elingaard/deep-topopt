import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cholesky
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

class SpCompFunction(Function):
    """Extension to pytorch library to calculate the element-wise compliance loss:
    L = u*K*u
    """
    @staticmethod
    def forward(ctx, rho, U, mesh, fea, penal):
        # detach so we can cast to NumPy
        rho, U = rho.detach(), U.detach()
        # compute compliance
        KE = torch.as_tensor(fea.KE,dtype=torch.float64).to(rho.device)
        ce = torch.sum(torch.matmul(U[mesh.edofMat],KE)*U[mesh.edofMat],axis=1)
        ce = ce.reshape((mesh.nelx,mesh.nely)).t() # column-wise reshape
        compliance = torch.sum((fea.Emin+rho**penal*(fea.Emax-fea.Emin))*ce)
        # save entries needed for backward pass
        ctx.save_for_backward(rho,ce)
        ctx.penal = penal
        ctx.fea = fea
        return compliance

    @staticmethod
    def backward(ctx, grad_output):
        """Given gradients from previous layer calculate the gradients
        of for the element-wise compliance. Gradients are a composite of:
        dL/(d w_i) = (dL/d rho_e)*(d rho_e)/(d w_i)
        dL/d rho_e) = (d u*Ku*)/(d rho_e) = -penal*rho_e^(p-1)*c_e
        (d rho_e)/(d w_i) = automatically calculated by autograd in next layer
        """
        # unpack input and set gradients to None
        grad_output = grad_output.detach()
        rho, ce = ctx.saved_tensors
        fea = ctx.fea
        penal = ctx.penal
        grad_rho = grad_U = grad_mesh = grad_fea = grad_penal = None
        grad_rho = (-penal*(fea.Emax-fea.Emin)*rho**(penal-1)*ce)*grad_output
        return grad_rho, grad_U, grad_mesh, grad_fea, grad_penal

def SpComp(rho, U, mesh, fea, penal):
    """Alias SpCompFunction class with the apply method"""
    return SpCompFunction.apply(rho, U, mesh, fea, penal)

class ComplianceLoss(nn.Module):
    """Class used to calculate the compliance loss:
    L = (u'Ku)/c_solid + lambda*||V||
    The loss is augmented by a penalization term on the volume constraint
    and normalized by the compliance of a fully solid design
    """

    def __init__(self,mesh,fea):
        super(ComplianceLoss, self).__init__()
        self.mesh = mesh
        self.fea = fea

    def compute_volume_violation(self,rho,vol_field):
        """
        Args:
        rho(nely x nelx float tensor): density array
        vol_field(nely x nelx float tensor): volume field

        Returns:
        volume_violation(float): violation of the volume constr
        """
        n_ele = len(self.mesh.IX)
        volfrac = torch.sum(vol_field)/n_ele
        volume_violation = torch.abs(torch.sum(rho)/n_ele-volfrac)
        return volume_violation

    def forward(self,rho,U,vol_field,solid_comp,penal,lambda_vol):
        """
        Args:
        rho(B x nely x nelx float tensor): density array
        U(B x nely x nelx float tensor): displacements array
        vol_field(B x nely x nely float tensor): volume field
        solid_comp(float): compliance for a fully solid design
        penal(float): SIMP penalization power
        lambda_vol(float): volume constraint penalization factor

        Returns:
        loss(B x 1 float tensor): loss for each entry in the batch
        compliance(B x 1 float tensor): compliance for each entry in the batch
        volume_violation(B x 1 float tensor): volume violation for each entry in the batch
        """

        batch_size = rho.shape[0]
        device = rho.device
        loss = torch.zeros(batch_size).to(device)
        compliance = torch.zeros(batch_size).to(device)
        volume_violation = torch.zeros(batch_size).to(device)
        # loop over each entry in the batch and solve the linear system in order calculate compliance
        for i in range(batch_size):
            compliance[i] = SpComp(rho[i],U[i],self.mesh,self.fea,penal)
            # calculate volume constraint
            volume_violation[i] = self.compute_volume_violation(rho[i],vol_field[i])
            # calculate total loss
            loss[i] = compliance[i]/solid_comp[i] + lambda_vol*volume_violation[i]
            # negative compliance usually indicates numerical over/underflow
            if compliance[i]<0:
                raise RuntimeError("Warning! Negative compliance")

        return loss,compliance,volume_violation


def batch_solve(fea,rho,fixed_dofs,load,penal,von_mises=False):
    """
    Solve the linear elasticity problem for a batch of density fields, boundary and load conditions

    Args:
    fea(object): finite element physics object
    rho(B x nely x nelx float tensor): density array
    fixed_dofs(B x N int tensor): fixed degrees-of-freedom
    load(B x ndof float tensor): load vector
    penal(float): SIMP penalization power
    von_mises(bool): whether to return Von Mises stresses or not

    Returns:
    displacements(list): list of tensors containing the displacements for each density field
    sigma_vm(list): list of tensors containing the von mises stresses for each density field
    """

    batch_size = rho.shape[0]
    device = rho.device
    fea.penal = penal
    rho = rho.detach().cpu().numpy()
    fixed_dofs = fixed_dofs.numpy()
    load = load.numpy()
    displacements = []
    von_mises = []
    for i in range(batch_size):
        fixed = fixed_dofs[i][fixed_dofs[i]>-1]
        fea.fixed_dofs = fixed
        fea.load = load[i]
        U = fea.solve_system(rho[i])
        U = torch.tensor(U).to(device)
        if von_mises==True:
            stress_tensor = self.fea.compute_element_stresses()
            sigma_vm = self.fea.compute_von_mises_stresses(stress_tensor)
            sigma_vm = normalize_data(sigma_vm)
            sigma_vm = torch.tensor(sigma_vm).to(device)
        else:
            sigma_vm=None

        displacements.append(U)
        von_mises.append(sigma_vm)

    return displacements,von_mises

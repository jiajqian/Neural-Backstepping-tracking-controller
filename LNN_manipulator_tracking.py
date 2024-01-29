from typing import Any
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from numpy.linalg import inv
import imageio
# import mediapy as media
# from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd.functional as AGF

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import time

import torch.optim as optim

from torch.autograd.functional import jacobian, hessian
from torchdiffeq import odeint as tor_odeint
from torchdiffeq import odeint_adjoint as tor_odeintadj
from functools import partial

from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import EarlyStopping
import math
import torch.linalg as linalg
# from scipy.integrate import odeint
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from numpy import cos, sin, arccos, arctan2, sqrt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mujoco_base import MuJoCoBase

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size": 20,
    "font.sans-serif": ["Helvetica"]})


class LNN(nn.Module):
    def __init__(self):
        super(LNN, self).__init__()
        self.coord_dim=3

        self.w_h0=nn.Parameter(torch.Tensor(self.coord_dim,32).uniform_(-0.1,0.1))
        self.w_y0=nn.Parameter(torch.Tensor(self.coord_dim,32).uniform_(-0.1,0.1))
        self.w_yu0=nn.Parameter(torch.Tensor(self.coord_dim,self.coord_dim).uniform_(-0.1,0.1))
        self.w_u0=nn.Parameter(torch.Tensor(self.coord_dim,32).uniform_(-0.1,0.1))
        self.w_h1=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_z1=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_zu1=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_y1=nn.Parameter(torch.Tensor(self.coord_dim,32).uniform_(-0.1,0.1))
        self.w_yu1=nn.Parameter(torch.Tensor(32,self.coord_dim).uniform_(-0.1,0.1))
        self.w_u1=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_h2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_z2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_zu2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_y2=nn.Parameter(torch.Tensor(self.coord_dim,32).uniform_(-0.1,0.1))
        self.w_yu2=nn.Parameter(torch.Tensor(32,self.coord_dim).uniform_(-0.1,0.1))
        self.w_u2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_z_out=nn.Parameter(torch.Tensor(32,1).uniform_(-0.1,0.1))
        self.w_zu_out=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_y_out=nn.Parameter(torch.Tensor(self.coord_dim,1).uniform_(-0.1,0.1))
        self.w_yu_out=nn.Parameter(torch.Tensor(32,self.coord_dim).uniform_(-0.1,0.1))
        self.w_u_out=nn.Parameter(torch.Tensor(32,1).uniform_(-0.1,0.1))

        self.b_y0=nn.Parameter(torch.rand(self.coord_dim))
        self.b_h0=nn.Parameter(torch.rand(32))
        self.b_0 =nn.Parameter(torch.rand(32))
        self.b_h1=nn.Parameter(torch.rand(32))
        self.b_y1=nn.Parameter(torch.rand(self.coord_dim))
        self.b_z1=nn.Parameter(torch.rand(32))
        self.b_1 =nn.Parameter(torch.rand(32))
        self.b_h2=nn.Parameter(torch.rand(32))
        self.b_z2=nn.Parameter(torch.rand(32))
        self.b_y2=nn.Parameter(torch.rand(self.coord_dim))
        self.b_2 =nn.Parameter(torch.rand(32))
        self.b_z_out=nn.Parameter(torch.rand(32))
        self.b_y_out=nn.Parameter(torch.rand(self.coord_dim))
        self.b_out=nn.Parameter(torch.rand(1))

        self.fc1 = nn.Linear(self.coord_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sp = nn.Softplus(beta=1)   #beta = 1 is steeper, so derivative = 1 is approached faster 

        
        self.apply(self._init_weights)

    def lagrangian(self, state): 
        u0=x = state[:self.coord_dim].clone()
        y = state[self.coord_dim:].clone()
        
        u1=self.sp(u0@self.w_h0+self.b_h0)
        z1=self.sp((y*(u0@self.w_yu0+self.b_y0))@self.w_y0+u0@self.w_u0+self.b_0)

        u2=self.sp(u1@self.w_h1+self.b_h1)
        z2=self.sp((z1*(u1@self.w_zu1+self.b_z1))@torch.relu(self.w_z1)+(y*(u1@self.w_yu1+self.b_y1))@self.w_y1+u1@self.w_u1+self.b_1)

        u3=self.sp(u2@self.w_h2+self.b_h2)
        z3=self.sp((z2*(u2@self.w_zu2+self.b_z2))@torch.relu(self.w_z2)+(y*(u2@self.w_yu2+self.b_y2))@self.w_y2+u2@self.w_u2+self.b_2)

        # u3=self.sp(u2@self.w_h2+self.b_h2)
        zout=self.sp((z3*(u3@self.w_zu_out+self.b_z_out))@torch.relu(self.w_z_out)+(y*(u3@self.w_yu_out+self.b_y_out))@self.w_y_out+u3@self.w_u_out+self.b_out)


        V=x
        V = self.sp(self.fc1(V))
        V = self.sp(self.fc2(V))
        V = self.sp(self.fc3(V))
        V = self.fc4(V)
        return zout-V   
    
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):                     #normalize weights according to paper
          self.fc1.weight.data.normal_(mean=0.0, std=2.2/sqrt(16))  #1st layer is 2.2/sqrt(n_1)
          self.fc2.weight.data.normal_(mean=0.0, std=0.58/sqrt(16)) #i-th layer is 0.58i/sqrt(n_i)
          self.fc3.weight.data.normal_(mean=0.0, std=0.58*2/sqrt(16))
          self.fc4.weight.data.normal_(mean=0.0, std=sqrt(16))    #last layer is sqrt(n)
          self.fc1.bias.data.zero_()
          self.fc2.bias.data.zero_()
          self.fc3.bias.data.zero_()
          self.fc4.bias.data.zero_()
    
    def forward(self, x):  


        x.requires_grad_()
        q=x[:self.coord_dim]
        dq=x[self.coord_dim:]

        a=torch.autograd.functional.hessian(self.lagrangian,x,create_graph=True)
        b=torch.autograd.functional.jacobian(self.lagrangian,x,create_graph=True)
        M=a[self.coord_dim:,self.coord_dim:]
        C=a[self.coord_dim:,:self.coord_dim]
        g=b[0,:self.coord_dim]
        ddq=(g-dq@C.t())@torch.inverse(M).t()

        dx=torch.cat([dq,ddq])
        dx.retain_grad()
        xd=x+0.001*dx
        return xd                        

    def t_forward(self, t, x):
        return self.forward(x)

    def get_system(self,state):
        dims=int(state.shape[0]/2)
        M=torch.zeros((dims,dims))
        dM_dt=torch.zeros((dims,dims))
        q=state[:dims].clone()
        dq=state[dims:].clone()
        dq.requires_grad_()
        q.requires_grad_()
        x=torch.cat((q,dq),dim=0)
        L=self.lagrangian(x)
        dL_ddq=torch.autograd.grad(L,dq,create_graph=True)[0]
        for i in range(dims):
            M[i,:]=torch.autograd.grad(dL_ddq[i],dq,create_graph=True)[0]
        for i in range(dims):
            for j in range(dims):
                dM_dt[i,j]=torch.autograd.grad(M[i,j],dq,create_graph=True)[0]@dq
        C=dM_dt-0.5*dM_dt.t()
        dL_dq=torch.autograd.grad(L,q,retain_graph=True)[0]

        g=0.5*dM_dt.t()@dq.t()-dL_dq


        return M,C,g

class Srelu(nn.Module):#定义Srelu
    def __init__(self) :
        super().__init__()

    def forward(self, z):
        z0_ = z.clone()
        d = torch.tensor(.01)
        z0_[(z0_>= d)] = z[(z0_>= d)]
        z0_[(z0_ <= 0.0)] = 0.0
        z0_[torch.logical_and(z0_ < d, z0_ > 0.0)] = z[torch.logical_and(z < d, z > 0.0)]**2/(2*d)
        return z0_

class FICNN(nn.Module):  # represents the controller gain
    def __init__(self):
        super(FICNN, self).__init__()
        self.w_z0=nn.Parameter(torch.Tensor(3,32).uniform_(-0.1,0.1))
        self.w_y1=nn.Parameter(torch.Tensor(3,32).uniform_(-0.1,0.1))
        self.w_y2=nn.Parameter(torch.Tensor(3,32).uniform_(-0.1,0.1))
        self.w_yout=nn.Parameter(torch.Tensor(3,1).uniform_(-0.1,0.1))
        self.w_z1=nn.Parameter(torch.Tensor(32,32).uniform_(0,0.1))
        self.w_z2=nn.Parameter(torch.Tensor(32,32).uniform_(0,0.1))
        self.w_zout=nn.Parameter(torch.Tensor(32,1).uniform_(0,0.1))
        self.srleu=Srelu()
        
    def forward(self, z):
        
        z0 = z.clone()
        z1 = z0 @ self.w_z0 
        z1s =self.srleu(z1)
        z2 = z1s @ torch.relu(self.w_z1)   +z0 @ self.w_y1 
        z2s = self.srleu(z2)
        z3 =  z2s @ torch.relu(self.w_z2)  + z0 @ self.w_y2
        z3s = self.srleu(z3)
        zout =  z3s @ torch.relu(self.w_zout)  + z0 @ self.w_yout
        zouts = self.srleu(zout)

        return zouts
 
class Damping(nn.Module):  # represents the controller gain
    def __init__(self):
        super(Damping, self).__init__()
        N = 3
        self.offdiag_output_dim = N*(N-1)//2
        self.diag_output_dim = N
        self.output_dim = self.offdiag_output_dim + self.diag_output_dim
        damp_min=torch.tensor([0.001,0.001,0.001])
        self.damp_min = damp_min
        self.w_d1=nn.Parameter(torch.Tensor(3,32).uniform_(-0.1,0.1))
        self.w_d2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_d3=nn.Parameter(torch.Tensor(32,self.diag_output_dim).uniform_(-0.1,0.1))
        self.w_o1=nn.Parameter(torch.Tensor(3,32).uniform_(-0.1,0.1))
        self.w_o2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.1))
        self.w_o3=nn.Parameter(torch.Tensor(32,self.offdiag_output_dim).uniform_(-0.1,0.1))
        self.b_d1=nn.Parameter(torch.zeros(32))
        self.b_d2=nn.Parameter(torch.zeros(32))
        self.b_d3=nn.Parameter(torch.zeros(self.diag_output_dim))
        self.b_o1=nn.Parameter(torch.zeros(32))
        self.b_o2=nn.Parameter(torch.zeros(32))
        self.b_o3=nn.Parameter(torch.zeros(self.offdiag_output_dim))

    def forward(self, input):

        x = input
        x0=x.clone()
        z=x.clone()

        d1 = x @ self.w_d1   + self.b_d1
        d1t = torch.tanh(d1)
        d2 =  d1t @ self.w_d2 + self.b_d2
        d2t = torch.tanh(d2)
        d3 = d2t @ self.w_d3  + self.b_d3
        d3r = (torch.relu(d3)+self.damp_min) * x
    


        n = self.diag_output_dim
        diag_idx = np.diag_indices(n)
        off_diag_idx = np.tril_indices(n, k=-1)
        D = torch.zeros(x.shape[0], n)

        o1 =  x @ self.w_o1 + self.b_o1
        o1t = torch.tanh(o1)
        o2 = o1t @ self.w_o2  + self.b_o2
        o2t = torch.tanh(o2)
        o3 = o2t @ self.w_o3  + self.b_o3



        for i in range(x.shape[0]):
            L = torch.zeros(n, n)
            diag_elements = d3r[i]
            off_diag_elements = o3[i]
            L[off_diag_idx] = off_diag_elements
            L[diag_idx] = diag_elements
            D_temp = L@L.t()
            D[i] = D_temp @ x[i]
        return D

class lnn_leaner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lnn=LNN()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam (self.lnn.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer,step_size=20, gamma=0.5)
        return [optimizer],[scheduler]
    
    def training_step(self, train_batch, batch_idx):
        num_traj = train_batch.size()[0]
        stage_cost=torch.zeros((num_traj-1,1))
        loss = torch.zeros(1).to(self.device)
        for i in range(num_traj-1):
            state=train_batch[i]
            target=train_batch[i+1]
            predict=self.lnn(state)            
            error=predict-target
            stage_cost[i]=1000*error.clone()@error.clone().t()

        loss=torch.mean(stage_cost)  

        self.log('train_loss', loss)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        num_traj = test_batch.size()[0]
        stage_cost=torch.zeros((num_traj,1))
        loss = torch.zeros(1).to(self.device)
        for i in range(num_traj-1):
            state=test_batch[i]
            target=test_batch[i+1]
            predict=self.lnn(state)
            error=predict-target
            stage_cost[i]=1000*error.clone()@error.clone().t()

        loss=torch.mean(stage_cost)  
        self.log('test_loss', loss)
        return loss

class controller_learner(pl.LightningModule):
    def __init__(self,lnn):
        super().__init__()
        coord_dim=3
        self._coord_dim = coord_dim
        self._state_dim = coord_dim*2
        self._action_dim = coord_dim
        self.init_quad_pot = 1.0,
        self.min_quad_pot = 1e-3,
        self.max_quad_pot = 1e1,
        self.FICNN_min_lr = 1e-1,
        self.alpha=torch.eye(coord_dim)*torch.tensor([0])

        init_quad_pot_param = torch.ones(coord_dim)*torch.Tensor([1.0])
        self._quad_pot_param = init_quad_pot_param
        self._min_quad_pot_param = torch.Tensor([1e-3])
        self._max_quad_pot_param = torch.Tensor([1e1])

        self.T=100
        self.ficnn_module=FICNN()
        self.damping_module=Damping()
        self.lnn=lnn
        T=self.T
        self.state = torch.zeros((T+1,6),requires_grad=True).to(self.device)
        self.state_error = torch.zeros((T,6),requires_grad=True).to(self.device)
        self.state_d = torch.zeros((T,9)).to(self.device)
        self.control = torch.zeros((T,3),requires_grad=True).to(self.device)
        self.control_wg = torch.zeros((T,3),requires_grad=True).to(self.device)
        self.stage_cost = torch.zeros((T+1),requires_grad=True).to(self.device)

        dt=0.01
        self.dt=torch.tensor(dt)

    def set_alpha(self, a):
        self.alpha=torch.eye(self.coord_dim)*torch.tensor([a])
        return self.alpha

    def phi(self, x):
        quad_pot = self._quad_pot_param.clamp(
            min=(self._min_quad_pot_param.item()),
            max=(self._max_quad_pot_param.item()))
        y=self.ficnn_module(x)+  (x @ torch.diag(quad_pot) @ x.t())

        return y

    def get_action(self, *inputs):
        state = torch.zeros((1,6), requires_grad=False).to(self.device)
        state[0] = inputs[0].clone()
        z1 = state[:,:3].requires_grad_() 
      
        psi = self.phi(z1)

        self.u_pot_1 = torch.autograd.grad(psi,z1,create_graph=True,retain_graph=True)
        self.u_pot = self.u_pot_1[0]
        z2=state[:,3:]+self.u_pot_1[0]
        self.u_dpot_1 = AGF.hessian(self.phi,z1,create_graph=True)
        self.u_dpot = self.u_dpot_1.clone().squeeze()
        self.u_damp = self.damping_module(z2)

        return self.u_pot,self.u_dpot, self.u_damp 

   
    def f(self,state,control):
        x=state.clone()
        u=control.clone()
        dq=state[3:].clone()
        M,C,G=self.lnn.get_system(x)
        ddq=(u-G-dq@C.t())@torch.inverse(M).t()

        # x.requires_grad_()
        # q=x[:self._coord_dim]
        # dq=x[self._coord_dim:]
        # a=torch.autograd.functional.hessian(self.lnn.lagrangian,x,create_graph=True)
        # b=torch.autograd.functional.jacobian(self.lnn.lagrangian,x,create_graph=True)
        # M=a[self._coord_dim:,self._coord_dim:]
        # C=a[self._coord_dim:,:self._coord_dim]
        # g=b[0,:self._coord_dim]
        # ddq=(u+g-dq@C.t())@torch.inverse(M).t()

        dx=torch.cat([dq,ddq])
        dx.retain_grad()
        state_t=x+self.dt*dx
        state_t.retain_grad()
        return state_t

    
    def forward(self, x):
        return 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params':self.ficnn_module.parameters()},
            {'params':self.damping_module.parameters()}], lr=1e-3)
        scheduler =StepLR(optimizer,step_size=40, gamma=0.5)
        return [optimizer],[scheduler]

    def training_step(self, train_batch, batch_idx):
        loss = torch.zeros((1)).to(self.device)
        T=self.T
        state = (self.state*torch.zeros(1)).to(self.device)
        state.retain_grad()
        state_error =(self.state_error *torch.zeros(1)).to(self.device)
        state_error.retain_grad()
        state_d =(self.state_d*torch.zeros(1)).to(self.device)
        control =(self.control*torch.zeros(1)).to(self.device)
        control.retain_grad()
        control_wg = (self.control_wg*torch.zeros(1)).to(self.device)
        control_wg.retain_grad()
        stage_cost =(self.stage_cost* torch.zeros(1)).to(self.device)
        stage_cost.retain_grad()
        state[0] = train_batch[0]
        for i in range(T):
            state_d[i] = torch.Tensor([sin(0.1*self.dt*i),cos(0.1*self.dt*i),sin(0.1*self.dt*i),0.1*cos(0.1*self.dt*i),-0.1*sin(0.1*self.dt*i),0.1*cos(0.1*self.dt*i),-0.1*0.1*sin(0.1*self.dt*i),-0.1*0.1*cos(0.1*self.dt*i),-0.1*0.1*sin(0.1*self.dt*i)])
            q_d = state_d[i,0:3]
            dq_d=state_d[i,3:6]
            ddq_d=state_d[i,6:]
            q=state[i,0:3]
            dq=state[i,3:6]
            self.get_action(state[i]-state_d[i,0:6])
            M,C,G=self.lnn.get_system(state[i])
            control[i] = G+(ddq_d-(dq-dq_d) @ self.u_dpot) @ M.t() + (dq_d-self.u_pot) @ C.t()-self.u_pot-self.u_damp             
            state[i+1] = self.f(state[i],control[i]).clone()
            state_error[i]=self.f(state[i],control[i])-state_d[i,:6]            
            stage_cost[i] =  (state_error[i,:3]).clone()@(state_error[i,:3]).clone().t()  
        loss = torch.sum(stage_cost)
        z0=torch.Tensor([0,0,0]).requires_grad_()   
        ddPhi_ddz0 = AGF.hessian(self.phi,z0,create_graph=True)
        ddPhi_ddz0= ddPhi_ddz0.clone().squeeze()
        A=self.alpha-ddPhi_ddz0.t() @ ddPhi_ddz0
        reg_value,reg_vector=torch.linalg.eig(A)
        real_reg=reg_value.real
        loss = loss + torch.sum(stage_cost)+torch.relu(real_reg.max()).sum()
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = torch.zeros((1)).to(self.device)
        T=self.T
        state = (self.state*torch.zeros(1)).to(self.device)
        state.retain_grad()
        state_error =(self.state_error *torch.zeros(1)).to(self.device)
        state_error.retain_grad()
        state_d =(self.state_d*torch.zeros(1)).to(self.device)
        control =(self.control*torch.zeros(1)).to(self.device)
        control.retain_grad()
        control_wg = (self.control_wg*torch.zeros(1)).to(self.device)
        control_wg.retain_grad()
        stage_cost =(self.stage_cost* torch.zeros(1)).to(self.device)
        stage_cost.retain_grad()
        state[0] = test_batch[0]
        for i in range(T):
            state_d[i] = torch.Tensor([sin(0.1*self.dt*i),cos(0.1*self.dt*i),sin(0.1*self.dt*i),0.1*cos(0.1*self.dt*i),-0.1*sin(0.1*self.dt*i),0.1*cos(0.1*self.dt*i),-0.1*0.1*sin(0.1*self.dt*i),-0.1*0.1*cos(0.1*self.dt*i),-0.1*0.1*sin(0.1*self.dt*i)])
            q_d = state_d[i,0:3]
            dq_d=state_d[i,3:6]
            ddq_d=state_d[i,6:]
            q=state[i,0:3]
            dq=state[i,3:6]
            self.get_action(state[i]-state_d[i,0:6])

            M,C,G=self.lnn.get_system(state[i])
            control[i] = G+(ddq_d-(dq-dq_d) @ self.u_dpot) @ M.t() + (dq_d-self.u_pot) @ C.t()-self.u_pot-self.u_damp             
            state[i+1] = self.f(state[i],control[i]).clone()
            state_error[i]=self.f(state[i],control[i])-state_d[i,:6]            
            stage_cost[i] =  (state_error[i,:3]).clone()@(state_error[i,:3]).clone().t()  
        loss = torch.sum(stage_cost)
        z0=torch.Tensor([0,0,0]).requires_grad_()   
        ddPhi_ddz0 = AGF.hessian(self.phi,z0,create_graph=True)
        ddPhi_ddz0= ddPhi_ddz0.clone().squeeze()
        A=self.alpha-ddPhi_ddz0.t() @ ddPhi_ddz0
        reg_value,reg_vector=torch.linalg.eig(A)
        real_reg=reg_value.real
        loss = loss + torch.sum(stage_cost)+torch.relu(real_reg.max()).sum()
        self.log('test_loss', loss)


class ManipulatorDrawing(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)
        self.simend = 200.0

        coord_dim=3
        self._coord_dim = coord_dim
        self._state_dim = coord_dim*2
        self._action_dim = coord_dim
        self.init_quad_pot = 1.0,
        self.min_quad_pot = 1e-3,
        self.max_quad_pot = 1e1,

        init_quad_pot_param = torch.ones(coord_dim)*torch.Tensor([1.0])
        self._quad_pot_param = init_quad_pot_param
        self._min_quad_pot_param = torch.Tensor([1e-3])
        self._max_quad_pot_param = torch.Tensor([1e1])

        self.lnn=LNN()
        self.ficnn_module=FICNN()
        self.damping_module=Damping()


    def reset(self):
        # Set initial angle of pendulum
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.0
        self.state= []
        self.state_error = []
        self.sim_time= []
        self.state_d = []
        
        
        # self.renderer.width=800
        # Set camera configuration
        self.cam.azimuth = 90
        self.cam.elevation = -15
        self.cam.distance = 8.0
        self.cam.lookat = np.array([0.0, 0.0, 3])

        mj.mj_forward(self.model, self.data)


        # mj.set_mjcb_control(self.controller)

    def set_NN(self,LNN_model,FICNN_model,damping_model):
        self.lnn=LNN_model
        self.ficnn_module=FICNN_model
        self.damping_module=damping_model
        return 0

    def phi(self, x):
        quad_pot = self._quad_pot_param.clamp(
            min=(self._min_quad_pot_param.item()),
            max=(self._max_quad_pot_param.item()))
        y=self.ficnn_module(x)+ (x @ torch.diag(quad_pot) @ x.t())

        return y
    
    def get_action(self, *inputs):
        state = torch.zeros((1,6), requires_grad=False)
        state[0] = inputs[0].clone()
        z1 = state[:,:3].requires_grad_() 
        psi = self.phi(z1)
        self.u_pot_1 = torch.autograd.grad(psi,z1,create_graph=True,retain_graph=True)
        self.u_pot = self.u_pot_1[0]
        z2=state[:,3:]+self.u_pot_1[0]
        self.u_dpot_1 = AGF.hessian(self.phi,z1)
        self.u_dpot = self.u_dpot_1.squeeze()
        self.u_damp = self.damping_module(z2)
        return self.u_pot,self.u_dpot, self.u_damp 

   
    def controller(self, model, data):
       
        state=torch.tensor(self.data.sensordata).float()
        q=state[:3]
        dq=state[3:6]
        self.sim_time.append(self.data.time)
        self.state.append(q.numpy())
        if len(self.state_d):
            state_error = self.state[-1]-self.state_d[-1]
            self.state_error.append(state_error)
        else:
            self.state_error.append(np.array([0,-1,0]))

        state_d = torch.Tensor([sin(0.1*self.data.time),cos(0.1*self.data.time),sin(0.1*self.data.time),0.1*cos(0.1*self.data.time),-0.1*sin(0.1*self.data.time),0.1*cos(0.1*self.data.time),-0.1*0.1*sin(0.1*self.data.time),-0.1*0.1*cos(0.1*self.data.time),-0.1*0.1*sin(0.1*self.data.time)])
        # set the desire state
        q_d = state_d[0:3]
        dq_d=state_d[3:6]
        ddq_d=state_d[6:9]
        self.state_d.append(q_d.numpy())
        # q=torch.tensor(self.data.qpos).float()
        # dq=torch.tensor(self.data.qvel).float()

        # MNN=np.zeros([3,3])
        # mj.mj_fullM(model,MNN,data.qM)
        # M=torch.Tensor(MNN)

        # C_force = torch.Tensor(data.qfrc_bias)
        # C=torch.pinverse(dq.reshape(1,3))@C_force.reshape(1,3)

        self.get_action(state[:6]-state_d[0:6])
        M,C,gc=self.lnn.get_system(state[:6])

        control =(ddq_d-(dq-dq_d) @ self.u_dpot) @ M.t()+(dq-self.u_pot)@C-self.u_pot-self.u_damp
        data.ctrl[0] = control[0,0].detach().numpy()
        data.ctrl[1] = control[0,1].detach().numpy()
        data.ctrl[2] = control[0,2].detach().numpy()


    def get_dataset(self):
        lnndata=[]
        while not glfw.window_should_close(self.window):
            simstart = self.data.time
            

            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                lnndata.append(self.data.sensordata[:6].copy())
                mj.mj_step(self.model, self.data)

            if self.data.time >= 10.0:
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            # fig = mj.MjvFigure()
            # mj.mjv_defaultFigure(fig)


            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            # mj.mjr_figure(viewport, fig, self.context)
            # frames.append(fig)
            # media.show_image(viewport)
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()
        return torch.tensor(lnndata).float()

    def gif(self):
        frames=[]
        while 1:
            simstart = self.data.time

            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                self.controller(self.model,self.data)
                mj.mj_step(self.model, self.data)
                self.renderer.update_scene(self.data,self.cam)
            if self.data.time >= self.simend:
                break
            # width=800
            # height=800
            fig=self.renderer.render()
            frames.append(fig.copy())
            
        imageio.mimsave("simulation_1.gif", frames, duration=1.0/60)

    def simulate(self):
        frames=[]
        while not glfw.window_should_close(self.window):
            simstart = self.data.time
            

            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                self.controller(self.model,self.data)
                mj.mj_step(self.model, self.data)

            if self.data.time >= self.simend:
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            # fig = mj.MjvFigure()
            # mj.mjv_defaultFigure(fig)


            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)
            # mj.mjr_figure(viewport, fig, self.context)
            # frames.append(fig)
            # media.show_image(viewport)
            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


        # imageio.imwrite("sample.gif", frames)

    def plot(self):
        _, ax=plt.subplots(2,1, sharex=True)
        sim_time=np.array(self.sim_time)
        state = np.array(self.state)
        state_d = np.array(self.state_d)
        state_error=np.array(self.state_error)

        np.save("mujoco_state_06_g_0.1t.npy",state)
        np.save("mujoco_state_error_06_g_0.1t.npy",state_error)
        # np.save("sim_time",sim_time)
        # state=np.load("mujoco_state_05.npy")
        # state_error=np.load("mujoco_state_error_05.npy")
        # sim_time=np.load("sim_time.npy")
        state_error_max=state_error[-70000:,0]**2+state_error[-70000:,1]**2+state_error[-70000:,2]**2
        print("state_error max:", state_error_max.max())
        ax[0].plot(sim_time, state[:,0],color='tab:blue',linewidth=1.5, label=r'${\beta_1}$')
        ax[0].plot(sim_time, state[:,1],color='tab:orange',linewidth=1.5, label=r'${\beta_2}$')
        ax[0].plot(sim_time, state[:,2],color='tab:green',linewidth=1.5, label=r'${\beta_3}$')
        
        ax[0].set_ylabel('robot link angles',fontsize=20)

        ax[1].plot(sim_time, state_error[:,0],color='tab:blue',linewidth=1.5, label=r'${\beta_1}-{\beta_1^{d}}$')
        ax[1].plot(sim_time, state_error[:,1],color='tab:orange',linewidth=1.5, label=r'${\beta_2}-{\beta_2^{d}}$')
        ax[1].plot(sim_time, state_error[:,2],color='tab:green',linewidth=1.5, label=r'${\beta_3}-{\beta_3^{d}}$')
        ax[1].set_ylabel('tracking errors',fontsize=20)
        
        
        ax[1].set_xlabel('time ',fontsize=20)

        ax[0].set_xlim(0,150)
        ax[1].set_xlim(0,150)

        ax[0].grid(linestyle='-')
        ax[1].grid(linestyle='-')
        # ax[2].plot(sim_time, state_d[:,0],color='tab:blue',linewidth=1.5, label=r'${\beta_1}$')
        # ax[2].plot(sim_time, state_d[:,1],color='tab:orange',linewidth=1.5, label=r'${\beta_2}$')
        # ax[2].set_ylabel(r'idael robot link angles')
        # ax[2].grid(linestyle='-')

        ax[0].legend()
        ax[1].legend()
        plt.savefig('final_mujoco_lnn_controller_train_0.1t.pdf',format='pdf')
        plt.show()


def main():
    xml_path = "MuJoCo\\threelink_manipulator.xml"
    sim = ManipulatorDrawing(xml_path)
    sim.reset()

    # # get dataset under gravity and no control input
    # train_data=sim.get_dataset()
    # np.save("lnn_traindata.npy",train_data.numpy())

    train_data=np.load("lnn_traindata.npy")
    train_data=torch.tensor(train_data)
    train_dataloader = DataLoader(train_data[:10000], batch_size=10)

    lnn_model=lnn_leaner()
    trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
                            callbacks=[], max_epochs=200)
    trainer.fit(lnn_model, train_dataloader)
    trainer.save_checkpoint("mujoco_lnn_threelink.ckpt")
    test_lnn_model=lnn_leaner.load_from_checkpoint(checkpoint_path="mujoco_lnn_threelink.ckpt")
    NBS_model=controller_learner(test_lnn_model.lnn)
    NBS_training_data = torch.Tensor([[0,0,0,0,0,0]])
    NBS_train_dataloader = DataLoader(NBS_training_data, batch_size=1)
    NBS_trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
                        callbacks=[], max_epochs=200)
    NBS_trainer.fit(NBS_model,NBS_train_dataloader)
    NBS_trainer.save_checkpoint("mujoco_controller.ckpt")

    test_NBS_model=controller_learner.load_from_checkpoint(checkpoint_path="mujoco_controller.ckpt",lnn=test_lnn_model.lnn)

    sim.reset()
    # # sim.model.opt.gravity=np.array([0,0,0])
    sim.set_NN(LNN_model=test_NBS_model.lnn,FICNN_model=test_NBS_model.ficnn_module,damping_model=test_NBS_model.damping_module)
    sim.gif()
    # sim.simulate()
    sim.plot()

if __name__ == "__main__":
    main()

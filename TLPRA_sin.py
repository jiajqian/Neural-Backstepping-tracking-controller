import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.autograd.functional as AGF
from pytorch_lightning.callbacks import EarlyStopping
import math
import torch.linalg as linalg
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
from numpy import cos, sin, arccos, arctan2, sqrt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch.optim.lr_scheduler import StepLR

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size":20,
    "font.sans-serif": ["Helvetica"]})


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


class ICNN(nn.Module):  # represents the controller gain
    def __init__(self):
        super(ICNN, self).__init__()
        self.w_z0=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.2))
        self.w_y1=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.2))
        self.w_y2=nn.Parameter(torch.Tensor(2,32).uniform_(-0.1,0.2))
        self.w_z1=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.2))
        self.w_z2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.1,0.2))
        self.w_yout=nn.Parameter(torch.Tensor(2,1).uniform_(-0.1,0.2))
        self.w_zout=nn.Parameter(torch.Tensor(32,1).uniform_(-0.1,0.2))

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
        N = 2
        self.offdiag_output_dim = N*(N-1)//2
        self.diag_output_dim = N
        self.output_dim = self.offdiag_output_dim + self.diag_output_dim
        damp_min=torch.tensor([0.001,0.001])
        self.damp_min = damp_min

        self.w_d1=nn.Parameter(torch.Tensor(2,32).uniform_(-0.2,0.2))
        self.w_d2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.2,0.2))
        self.w_d3=nn.Parameter(torch.Tensor(32,2).uniform_(-0.2,0.2))
        self.w_o1=nn.Parameter(torch.Tensor(2,32).uniform_(-0.2,0.2))
        self.w_o2=nn.Parameter(torch.Tensor(32,32).uniform_(-0.2,0.2))
        self.w_o3=nn.Parameter(torch.Tensor(32,1).uniform_(-0.2,0.2))

        self.b_d1=nn.Parameter(torch.zeros(32))
        self.b_d2=nn.Parameter(torch.zeros(32))
        self.b_d3=nn.Parameter(torch.zeros(2))
        self.b_o1=nn.Parameter(torch.zeros(32))
        self.b_o2=nn.Parameter(torch.zeros(32))
        self.b_o3=nn.Parameter(torch.zeros(1))

    def forward(self, input):

        x0 = input
        x=x0.clone()
        z=x0.clone()

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



class NBS_tracking_control_learner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        coord_dim=2
        self._coord_dim = coord_dim
        self._state_dim = coord_dim*2
        self._action_dim = coord_dim
        self.init_quad_pot = 1.0,
        self.min_quad_pot = 1e-3,
        self.max_quad_pot = 1e1,
        self.icnn_min_lr = 1e-1,
        self.alpha=0

        self.S = torch.eyes(coord_dim)*torch.Tensor([1.0])

        self.icnn_module=ICNN()
        self.damping_module=Damping()
        self.T=100
        self.state = torch.zeros((self.T+1,4),requires_grad=True).to(self.device)
        self.state_error = torch.zeros((self.T,4),requires_grad=True).to(self.device)
        self.state_d = torch.zeros((self.T,6)).to(self.device)
        self.control = torch.zeros((self.T,2),requires_grad=True).to(self.device)
        self.control_wg = torch.zeros((self.T,2),requires_grad=True).to(self.device)
        self.stage_cost = torch.zeros((self.T+1),requires_grad=True).to(self.device)

        self.dt=torch.tensor(0.01)
        self.m1=torch.tensor(1.0)
        self.m2=torch.tensor(1.0)
        self.l1=torch.tensor(1.0)
        self.l2=torch.tensor(1.0)
        self.g = torch.tensor(9.8)
        self.flag = 0
        
    def set_alpha(self, a=0):
        self.alpha=torch.eye(2)*torch.tensor([a])
        return self.alpha

    def phi(self, x):
        y=self.icnn_module(x)+  (x @ self.S @ x.t())
        return y

    def get_action(self, *inputs):
        state = torch.zeros((1,4), requires_grad=False).to(self.device)
        state[0] = inputs[0].clone()
        z1 = state[:,:2].requires_grad_() 
      
        psi = self.phi(z1)

        self.u_pot_1 = torch.autograd.grad(psi,z1,create_graph=True,retain_graph=True)
        self.u_pot = self.u_pot_1[0]
        z2=state[:,2:]+self.u_pot_1[0]

        self.u_dpot_1 = AGF.hessian(self.phi,z1,create_graph=True)
        self.u_dpot = self.u_dpot_1.clone().squeeze()
        self.u_damp = self.damping_module(z2)
       
        return self.u_pot,self.u_dpot, self.u_damp 
    def Gravity(self, x):
        
        b_2=x[1].clone()
        M=torch.tensor([[(self.m1+self.m2)*self.l1**2+self.m2*self.l2**2+2*self.m2*self.l1*self.l2*torch.cos(b_2),self.m2*self.l2**2+self.m2*self.l2*self.l1*torch.cos(b_2)],
                                    [self.m2*self.l1*self.l2*torch.cos(b_2)+self.m2*self.l2**2,self.m2*self.l2**2]])
        return M
    
    def Correlation(self,state):
        b_1=state[0].clone()
        b_2=state[1].clone()
        v_1=state[2].clone()
        v_2=state[3].clone()
        C=torch.tensor([[-self.m2*self.l1*self.l2*v_2*torch.sin(b_2),-self.m2*self.l1*self.l2*(v_1+v_2)*torch.sin(b_2)],
                        [self.m2*self.l1*self.l2*v_1*torch.sin(b_2),0]])
        return C

    def gravity_compensate(self,x):
        b1=x[0].clone()
        b2=x[1].clone()
        gc=torch.tensor([torch.tensor(2)*self.g*torch.cos(b1)+self.g*torch.cos(b1+b2),self.g*torch.cos(b1+b2)])

        return gc
    
    def f(self,state,control):
        x=state.clone()
        u=control.clone()
        dis=torch.tensor([1,1])
        # dis=2*torch.rand(1,2)-1
        b_1=x[0]
        b_2=x[1]
        v_1=x[2]
        v_2=x[3]
        u=u#+dis
        u=u.unsqueeze(0)
        M=self.Gravity(x)
        V=torch.tensor([[-self.m2*self.l1*self.l2*(2*v_1*v_2+v_2**2)*torch.sin(b_2)],[self.m2*self.l1*self.l2*(v_1**2)*torch.sin(b_2)]])
        G=torch.tensor([[(self.m1+self.m2)*self.g*self.l1*torch.cos(b_1)+self.m2*self.g*self.l2*torch.cos(b_1+b_2)],[self.m2*self.g*self.l2*torch.cos(b_1+b_2)]])
        ux=u.t()
        dot=torch.inverse(M) @ (ux-V-G)
        dx=torch.cat((x[2:],dot.t().squeeze()),0)    
   
        state_t=x+self.dt*dx
        state_t.retain_grad()
        return state_t

    def forward(self, x):
        return 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params':self.icnn_module.parameters()},
            {'params':self.damping_module.parameters()}], lr=1e-3)
        scheduler =StepLR(optimizer,step_size=40, gamma=0.5)
        return [optimizer],[scheduler]

    def training_step(self, train_batch, batch_idx):
        loss = torch.zeros((1)).to(self.device)
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
        for i in range(self.T):
            state_d[i] = torch.Tensor([sin(0.1*self.dt*i),cos(0.1*self.dt*i),0.1*cos(0.1*self.dt*i),-0.1*sin(0.1*self.dt*i),-0.1*0.1*sin(0.1*self.dt*i),-0.1*0.1*cos(0.1*self.dt*i)])
            q_d = state_d[i,0:2]
            dq_d=state_d[i,2:4]
            ddq_d=state_d[i,4:6]
            q=state[i,0:2]
            dq=state[i,2:4]
            self.get_action(state[i]-state_d[i,0:4])
            gc=self.gravity_compensate(state[i])
            C=self.Correlation(state[i])
            M=self.Gravity(state[i])
            control[i] = gc +(ddq_d-(dq-dq_d) @ self.u_dpot) @ M.t() + (dq_d-self.u_pot) @ C.t()-self.u_pot-self.u_damp 
            state[i+1] = self.f(state[i],control[i]).clone()
            state_error[i]=self.f(state[i],control[i])-state_d[i,:4]
            
            stage_cost[i] =  (state_error[i,:2]).clone()@(state_error[i,:2]).clone().t()  
        z0=torch.Tensor([0,0]).requires_grad_()   
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
        T=5000
        state = torch.zeros((T+1,4)).to(self.device)
        state_error = torch.zeros((T,4)).to(self.device)
        state_d = torch.zeros((T,4)).to(self.device)
        control = torch.zeros((T,2)).to(self.device)
        stage_cost = torch.zeros((T+1)).to(self.device)
        state[0,0:4] = test_batch
        for i in range(T):
            state_d[i] = torch.Tensor([sin(0.1*self.dt*i),cos(0.1*self.dt*i),0.1*cos(0.1*self.dt*i),-0.1*sin(0.1*self.dt*i),-0.1*0.1*sin(0.1*self.dt*i),-0.1*0.1*cos(0.1*self.dt*i)])            # state_d[t] = torch.Tensor([1,1,0,0,0,0])
            q_d = state_d[i,0:2]
            dq_d=state_d[i,2:4]
            ddq_d=state_d[i,4:6]
            q=state[i,0:2]
            dq=state[i,2:4]
            self.get_action(state[i]-state_d[i,0:4])
            gc=self.gravity_compensate(state[i])
            C=self.Correlation(state[i])
            M=self.Gravity(state[i])
            control[i] = gc +(ddq_d-(dq-dq_d) @ self.u_dpot) @ M.t() + (dq_d-self.u_pot) @ C.t()-self.u_pot-self.u_damp 
            state[i+1] = self.f(state[i],control[i])
            state_error[i]=state[i+1]-state_d[i,:4]
            stage_cost[i] = (state_error[i]).clone() @ (state_error[i]).clone().t() 
        z0=torch.zeros([0,0])   
        ddPhi_ddz0 = AGF.hessian(self.phi,z0,create_graph=True)
        ddPhi_ddz0= ddPhi_ddz0.clone().squeeze()
        A=self.alpha-ddPhi_ddz0.t() @ ddPhi_ddz0
        reg_value,reg_vector=torch.linalg.eig(A)
        real_reg=reg_value.real
        loss = loss + torch.sum(stage_cost)+torch.relu(real_reg.max()).sum()
        self.log('test_loss', loss)
    
    def caculate_stable_error(self):
        T=20000
        state = torch.zeros(T+1,4).to(self.device)
        control = torch.zeros(T,2).to(self.device)
        state_d = torch.zeros(T,6).to(self.device)
        state_error = torch.zeros(T,4).to(self.device)
        # stage_cost = torch.zeros(T+1).to(self.device)
        state[0] = torch.tensor([0,0,0,0])
        time = torch.zeros(T,1).to(self.device)
        for t in range(T):
            state_d[t] = torch.Tensor([sin(0.1*self.dt*t),cos(0.1*self.dt*t),0.1*cos(0.1*self.dt*t),-0.1*sin(0.1*self.dt*t),-0.1*0.1*sin(0.1*self.dt*t),-0.1*0.1*cos(0.1*self.dt*t)])
            # state_d[t] = torch.Tensor([1,1,0,0,0,0])
            q_d = state_d[t,0:2]
            dq_d=state_d[t,2:4]
            ddq_d=state_d[t,4:6]
            q=state[t,0:2]
            dq=state[t,2:4]


            self.get_action(state[t]-state_d[t,0:4])
            gc=self.gravity_compensate(state[t])
            C=self.Correlation(state[t])
            M=self.Gravity(state[t])
            control[t] = gc +(ddq_d-(dq-dq_d) @ self.u_dpot) @ M.t() + (dq_d-self.u_pot) @ C.t()-self.u_pot-self.u_damp #-self.u_quad
            time[t]=torch.tensor(self.dt*t)
            
            state[t+1] = self.f(state[t],control[t])
            state_error[t]=state[t+1]-state_d[t,:4]
        z1= state_error[-7000:,0]**2+state_error[-7000:,1]**2
        z1_max = z1.max().detach().numpy() 
        print(z1_max)
        return z1_max
    
    def plottrajetory(self):
        T=20000 #sim time
        state = torch.zeros(T+1,4).to(self.device)
        control = torch.zeros(T,2).to(self.device)
        state_d = torch.zeros(T,6).to(self.device)
        state_error = torch.zeros(T,4).to(self.device)
        state[0] = torch.tensor([0,0,0,0])
        time = torch.zeros(T,1).to(self.device)
        for t in range(T):
            state_d[t] = torch.Tensor([sin(0.1*self.dt*t),cos(0.1*self.dt*t),0.1*cos(0.1*self.dt*t),-0.1*sin(0.1*self.dt*t),-0.1*0.1*sin(0.1*self.dt*t),-0.1*0.1*cos(0.1*self.dt*t)])
            # state_d[t] = torch.Tensor([1,1,0,0,0,0])
            q_d = state_d[t,0:2]
            dq_d=state_d[t,2:4]
            ddq_d=state_d[t,4:6]
            q=state[t,0:2]
            dq=state[t,2:4]

            self.get_action(state[t]-state_d[t,0:4])
            gc=self.gravity_compensate(state[t])
            C=self.Correlation(state[t])
            M=self.Gravity(state[t])
            control[t] = gc +(ddq_d-(dq-dq_d) @ self.u_dpot) @ M.t() + (dq_d-self.u_pot) @ C.t()-self.u_pot-self.u_damp #-self.u_quad
            time[t]=torch.tensor(self.dt*t)
            
            state[t+1] = self.f(state[t],control[t])
            state_error[t]=state[t+1]-state_d[t,:4]
        z1= state_error[-7000:,0]**2+state_error[-7000:,1]**2
        z1_max = z1.max().detach().numpy() 
        print("stable tracking error:", z1_max)
        np.save("state_train_1.npy",state.clone().detach().numpy())
        np.save("state_error_train_1.npy",state_error.clone().detach().numpy())
        np.save("time.npy",time.clone().detach().numpy())
        plt.figure()
        fig,ax= plt.subplots(2,1,sharex=True)
        ax[1].set_xlabel('time ', fontsize=20)
        ax[0].set_ylabel('robot link angles ',fontsize=20)
        ax[1].set_ylabel('tracking errors ',fontsize=20)


        ax[0].grid(linestyle='-')
        
        ax[0].plot(time[:T],state[:T,0].detach(), color='tab:blue',linewidth=1.5, label=r'$\beta_1$')
        ax[0].plot(time[:T],state[:T,1].detach(), color='tab:orange',linewidth=1.5, label=r'$\beta_2$')
        ax[1].grid(linestyle='-')
        ax[1].plot(time[:T],state_error[:T,0].detach(), color='tab:blue',linewidth=1.5, label=r'$\beta_1$-$\beta_1^{d}$')
        ax[1].plot(time[:T],state_error[:T,1].detach(), color='tab:orange',linewidth=1.5, label=r'$\beta_2$-$\beta_2^{d}$')

        ax[0].legend()
        ax[1].legend()
        plt.savefig('train_TLPRA_0.1sincos_trajectories.pdf', format='pdf',bbox_inches="tight")
        
        # plt.show()
        return state

if __name__ == '__main__':
        seed=3
        pl.seed_everything(seed)
        training_data = torch.Tensor([[0,0,0,0]])
        train_dataloader = DataLoader(training_data, batch_size=1)

        train_model=NBS_tracking_control_learner()
        train_model.plottrajetory()
        # train_model.caculate_stable_error()
        trainer = pl.Trainer(accelerator="cpu", num_nodes=1,
                            callbacks=[], max_epochs=200)
        trainer.fit(train_model,train_dataloader)
        trainer.save_checkpoint("2link_0.1sincos.ckpt")

        test_model=NBS_tracking_control_learner.load_from_checkpoint("2link_0.1sincos.ckpt")
        test_model.plottrajetory()
        # test_model.caculate_stable_error()

        ### different alpha in training contrast
        
        
        # alpha=np.linspace(0.05,2,40)
        
        # training_data = torch.Tensor([[0,0,0,0]])
        # train_dataloader = DataLoader(training_data, batch_size=1)
        
  
        # error_max=[]
        # trainers=locals()
        # models = locals()
        # for a in alpha:
        #     pl.seed_everything(seed)
        #     models['model_'+str(a)] = NBS_tracking_control_learner()
        #     models['model_'+str(a)].set_alpha(a)
        #     # models['model_'+str(a)].caculate_stable_error()
        #     trainers['trainer_'+str(a)] = pl.Trainer(accelerator="cpu", num_nodes=1,
        #                     callbacks=[], max_epochs=200)
        #     trainers['trainer_'+str(a)].fit(models['model_'+str(a)], train_dataloader)
        #     trainers['trainer_'+str(a)].save_checkpoint("NBS_1_final_2link_1_alpha_0.1_4_40_"+str(a)+".ckpt")
        #     print(a,"trianing complete----\n")

        #     # models['model_'+str(a)].caculate_stable_error()
        #     # print("before test------\n")
        #     # model = NBS_tracking_control_learner()
        #     # model.set_alpha(a)
        #     # trainer.fit(model, train_dataloader)
            
        #     train_model=NBS_tracking_control_learner().load_from_checkpoint(
        #     checkpoint_path="NBS_1_final_2link_1_alpha_0.1_4_40_"+str(a)+".ckpt")
        #     error_max.append(train_model.caculate_stable_error())

        # error_alpha=np.array(error_max)
        # np.save('error_alpha.npy',error_alpha)

        # test_model=NBS_tracking_control_learner().load_from_checkpoint(
        #     checkpoint_path="NBS_1_tracking_model_2link_1_alpha_u_0_4_11_01_4.0.ckpt")
        # test_model.plottrajetory()

        # error_alpha = np.load('error_alpha.npy')
        # bound = 0.5*2/(alpha**2)
        # z1=error_alpha
        # _, ax=plt.subplots(1,1, sharex=True)
        # ax.scatter(alpha, z1,color='tab:red', label=r'${\||z_1\||^2}$')
        # ax.plot(alpha, bound,color='tab:blue',linewidth=1.5, label='bound')
        # # ax.plot(alpha, error_alpha[:,0],color='tab:blue',linewidth=1.5, label=r'${\beta_1-\beta_1^{desire}}$')
        # # ax.plot(alpha, error_alpha[:,1],color='tab:orange',linewidth=1.5, label=r'${\beta_2-\beta_2^{desire}}$')
        # ax.set_ylabel('maximal steady-state tracking error',fontsize=20)
        # ax.set_xlabel(r'${\alpha}$',fontsize=20)
        # # ax.set_ylim(0,0.5)
        # ax.grid(linestyle='-')
        # ax.legend()

        # axins = inset_axes(ax, width='50%',height='30%',loc='center right')
        # axins.plot(alpha[1:], bound[1:],color='tab:blue')
        # axins.scatter(alpha[1:], z1[1:],color='tab:red')
        # axins.set_xlim(1.5,2)
        # axins.set_ylim(0,0.5)
        # axins.spines['top'].set_color('red')
        # axins.spines['right'].set_color('red')
        # axins.spines['bottom'].set_color('red')
        # axins.spines['left'].set_color('red')
        # axins.plot([8,10,10,4,4],[0,0,0.1,0.1,0], 'red')
        # # plt.show()
        # plt.savefig('alpha_error_1_0.05_2_40_1D_bound.pdf', format='pdf',bbox_inches="tight")

  
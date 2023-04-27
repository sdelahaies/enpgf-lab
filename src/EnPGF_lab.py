import numpy as np
import random 
import torch
from torch import matmul
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.distributions.gamma import Gamma
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def dN_to_sparse_csv(dN,fname=None):
    dN_sparse = dN.to_sparse_coo()
    dN_indices = dN_sparse.indices().cpu().numpy()
    dN_values = dN_sparse.values().cpu().numpy() 
    data= np.array([dN_indices[0,:].transpose(),
                    dN_indices[1,:].transpose(),
                    dN_values])
    df = pd.DataFrame(data.transpose(),columns=["index_1","index_2","values"]).astype('uint32')
    compression = 'gzip'
    #compression = ''
    fname = f'dN_{dN.shape[0]}_{dN.shape[1]}.csv.{compression}'
    df.to_csv(fname,index=False,compression='gzip')
    #df.to_csv(fname,index=False)

def dN_load(fname,shape=None,compression=None):
    if shape is None:
        shape = (int(fname.split('_')[1]),
                 int(fname.split('_')[2].split('.')[0]))
    if compression is None:
        compression = fname.split('.')[-1]
    df = pd.read_csv(fname,compression=compression)    
    dN = torch.sparse_coo_tensor(df[["index_1","index_2"]].to_numpy().transpose(),df["values"].to_numpy(),shape)
    return dN.to_dense(),shape[1],shape[0]

def set_seed(seed=None, seed_torch=True):
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  print(f'Random seed {seed} has been set.')

set_seed(seed=42)

# def random_gamma(shape, a, b=1.0,device='cpu'):
#   alpha = torch.ones(shape).to(device) * a+0.000001
#   beta = torch.ones(shape) * torch.tensor(b)
#   gamma_distribution = Gamma(alpha, beta.to(device))
#   return gamma_distribution.sample().to(device)

def init_example():
    Nnodes=7
    alphat=np.zeros([Nnodes,Nnodes])
    alphat[0:3,0:3]=0.5
    alphat[3:6,3:6]=0.4
    alphat[3:5,3:5]=0.4
    for i in range(Nnodes):
        alphat[i,i]=alphat[i,i]+0.2
    nout=int(np.floor(Nnodes*Nnodes/8)); tmp=np.random.randint(0,Nnodes-1,[2,nout])
    for i in range(nout):
                alphat[tmp[0,i],tmp[1,i]]=alphat[tmp[0,i],tmp[1,i]]+0.4*np.random.gamma(1,0.5)
    #plt.matshow(alphat)
    #plt.show()
    mut=np.random.gamma(2,0.3,[Nnodes])
    betat=10
    return mut,betat,alphat
    
def gen_data(dt,mu,alpha,beta,Nstep,N):
    dNall=[]
    dN_list=[]
    rad=np.sort(np.linalg.eig(alpha/beta)[0])[-1]
    if rad>1:
        print('spectral radius >1')
        print('aborting')
        return np.array(dNall),dN_list
    print(np.sort(np.linalg.eig(alpha/beta)[0])[-1])
    lamb=mu.copy()
    for _ in range(Nstep):
        dN_now=np.random.poisson(lamb*dt)
        dNall.append(dN_now)
        lamb = mu+(lamb-mu)*(1-beta*dt)+np.dot(alpha,dN_now)
        lamb[lamb<=0.]=0.000000001
    return np.array(dNall),dN_list

class EnPGF():
    def __init__(self,n,nens,dN,dt,device='cpu',inflation=1,init_alpha=0.2):
        tmp = np.random.gamma(1.,1,[nens,n])
        tmp0 = np.random.normal(3,0.5,[nens,n])
        tmp2=np.random.gamma(1.,0.1,[n,n,nens])
        
        self.device = device
        self.n =torch.tensor(n).to(device)
        self.nens = torch.tensor(nens).to(device)
        self.lb = torch.from_numpy(tmp.copy()).to(device)
        self.mu = torch.from_numpy(np.log(tmp.copy())).to(device)
        self.beta = torch.from_numpy(np.log(tmp0.copy())).to(device)
        self.alpha = torch.from_numpy(np.log(init_alpha*tmp2.copy())).to(device)
        self.dN = torch.tensor(dN).to(device)
        self.dt = torch.tensor(dt).to(device)
        self.inflation=torch.tensor(inflation).to(device)
        self.nstep=torch.tensor(0).to(device)
    
    def __random_gamma(self,shape, a, b=1.0):
        alpha = torch.ones(shape).to(self.device)* a+torch.tensor(0.000001).to(self.device)
        beta = torch.ones(shape) * torch.tensor(b)
        gamma_distribution = Gamma(alpha, beta.to(self.device))
        return gamma_distribution.sample().to(self.device)
            
    def __analysis(self,yo):
        mask1 = yo == 0
        mask2 = mask1.unsqueeze(0).expand([self.nens,self.n])
        lbf = self.dt * self.lb.clone().detach()
        lbdt = self.dt * self.lb.clone().detach()
        lb_bar = lbdt.mean(0)
        A = lbdt-lb_bar
        Vf = self.inflation*torch.einsum('ia,ia->a',A,A)/(self.nens-1)
        lb_bar_a = lb_bar+(yo-lb_bar)*Vf/(lb_bar+Vf)
        K1 = A/lb_bar
        lb_tmp = self.__random_gamma(shape=(self.nens,self.n), a=yo).to(self.device)
        lb_tmp_bar = lb_tmp.mean(0)
        K2=lb_tmp-lb_tmp_bar
        K2/=lb_tmp_bar
        Vfr=Vf/lb_bar**2
        K2 = torch.where(mask2,K1,K2)
        tmp_var = torch.where(mask1,1.,Vfr/(Vfr+1./yo))
        A_a = K1+ tmp_var*(K2-K1)
        A_a *=lb_bar_a
        lb_a = lb_bar_a+A_a
        self.lb = lb_a/self.dt
        kk = (lb_a -lbf)/Vf
        return kk, A
    
    #@torch.jit.script
    def __update_mu(self,kk,A):
        Ak = self.mu-self.mu.mean(0)
        Cf = torch.einsum('ia,ia->a',A,Ak)/(self.nens-1)
        self.mu+=Cf*kk
            
    #@torch.jit.script
    def __update_beta(self,kk,A):
        Ak = self.beta-self.beta.mean(0)
        Cf = torch.einsum('ia,ia->a',A,Ak)/(self.nens-1)
        self.beta+=Cf*kk
        
    #@torch.jit.script
    def __update_alpha(self, kk, A):
        a_tmp = self.alpha.reshape([self.nens,self.n*self.n])
        Ak = a_tmp-a_tmp.mean(0)
        A= A.repeat([1,self.n])
        Cf = torch.einsum('ia,ia->a',A,Ak)/(self.nens-1)
        kk= kk.repeat([1,self.n])
        a_tmp += Cf*kk
        self.alpha = a_tmp.reshape([self.nens,self.n,self.n])
    
    #@torch.jit.script
    def __forecast(self,y):
        alpha = self.alpha.permute(2,0,1)
        AdN = matmul(torch.exp(alpha),y.double())
        tmp3 = torch.exp(self.mu) + (1-self.dt*torch.exp(self.beta)) * (self.lb - torch.exp(self.mu))
        lamb = tmp3+AdN.t()
        lamb[lamb<=0.]= 0.00000001
        self.lb = lamb
   
    def reset(self):
        tmp = np.random.gamma(1.,1,[self.nens])
        tmp0 = np.random.normal(3,0.5,[self.nens])
        tmp2=np.random.gamma(1.,0.1,[self.nens,self.n])
        for i in range(self.n):
            if self.lb.mean(0)[i].isnan():
                self.lb[:,i]=torch.from_numpy(tmp.copy()).to(self.device)
                self.mu[:,i]=torch.from_numpy(np.log(tmp.copy())).to(self.device)
                self.beta[:,i]=torch.from_numpy(np.log(tmp0.copy())).to(self.device)
                self.alpha[:,:,i]=torch.from_numpy(np.log(0.2*tmp2.copy())).to(self.device)
        
          
    def train(self,nstep=None,reset=False):
        if nstep is None:
            nstep = len(self.dN)
        with torch.cuda.amp.autocast():
            with torch.no_grad(): 
                for i in tqdm(range(nstep)):
                    kk,A = self.__analysis(self.dN[i,:])
                    self.__update_mu(kk,A)
                    self.__update_beta(kk,A)
                    self.__update_alpha(kk,A)
                    self.__forecast(self.dN[i,:])
                    if reset and i%500==0:
                        self.reset()
        self.nstep += nstep
        
    def display_alpha(self,cname='inferno',truth=None):
        if truth:
            alphat= np.loadtxt(truth,delimiter=",",dtype=float)
            fig,(ax1,ax2)=plt.subplots(1,2)
            ax1.matshow(alphat,cmap=mpl.colormaps[cname])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.matshow(np.exp(self.alpha.mean(0).cpu().numpy().transpose()),cmap=mpl.colormaps[cname])
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            plt.matshow(np.exp(self.alpha.mean(0).cpu().numpy().transpose()),cmap=mpl.colormaps[cname])
        plt.show()

    def save(self,prefix='',outname='enpgf.json',alphaname='alpha.csv.gz',dataname='data.csv.gz'):
        dict_out={
            "enpgf":
                {
                    "fname":prefix+outname,
                    'n':self.n.cpu().numpy().tolist(),
                    'nens':self.nens.cpu().numpy().tolist(),
                    'nstep':self.nstep.cpu().numpy().tolist(),
                    'dN':prefix+dataname,
                    'alpha':prefix+alphaname
                }
        }
        with open(prefix+outname, "w") as write_file:
            json.dump(dict_out, write_file, indent=4)
        alpha = np.exp(self.alpha.mean(0).cpu().numpy().transpose())
        np.savetxt(prefix+alphaname, alpha, delimiter=',')
            

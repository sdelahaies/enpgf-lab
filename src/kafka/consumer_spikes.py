from kafka import KafkaConsumer
from json import loads
from time import sleep

import numpy as np
import random 
import torch
from torch import matmul
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributions.gamma import Gamma
import pandas as pd
import yaml

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json

def write_json(new_data, filename='enpgf.json'):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        file_data["EnPGF_trainer"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent = 4)

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
        #self.dN = torch.tensor(dN).to(device)
        self.dt = torch.tensor(dt).to(device)
        self.inflation=torch.tensor(inflation).to(device)
    
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
    
    def train_step(self,dN_step):
        dN = torch.tensor(dN_step).to(device)
        with torch.cuda.amp.autocast():
            with torch.no_grad(): 
                kk,A = self.__analysis(dN)
                self.__update_mu(kk,A)
                self.__update_beta(kk,A)
                self.__update_alpha(kk,A)
                self.__forecast(dN)
  
    
    
    def display_alpha(self):
        plt.matshow(np.exp(self.alpha.mean(0).cpu().numpy().transpose()))
        plt.show()

with open("src/kafka/enpgf.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile,Loader=yaml.Loader)

n = cfg["enpgf"]["n"]
nens = cfg["enpgf"]["nens"]
dt = cfg["enpgf"]["dt"] 
device = cfg["enpgf"]["device"]
topic = cfg["enpgf"]["topic"]

enpgf= EnPGF(n=n,nens=nens,dN=None,dt=dt,device=device)

consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['kafka0:29092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='spikes-activity',
    value_deserializer=lambda x: loads(x.decode('utf-8')))


for event in consumer:
    event_data = event.value
    dN_step = np.array(event_data["spikes"])
    enpgf.train_step(dN_step)
    data = np.round(enpgf.alpha.cpu().numpy().mean(0),2)
    print('kafka consumer training step', event_data["step"])
    #if event_data["step"]%500 == 0:
    #    output=open("alpha.txt",'wb')
    #    output.write(data.tobytes())    

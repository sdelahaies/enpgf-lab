from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
import yaml

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

def init_alpha(Nnodes):
    alphat=np.zeros([Nnodes,Nnodes])
    alphat[0:3,0:3]=0.4
    alphat[3:7,3:7]=0.4
    alphat[15:20,15:20]=0.3
    alphat[30:40,30:40]=0.3
    alphat[15:25,85:95]=0.2
    alphat[85:95,15:25]=0.2
    #alphat[120:135,120:135]=0.3
    for i in range(Nnodes):
        alphat[i,i]=alphat[i,i]+0.2
    #alphat[280:,280:]=0.2
    #alphat[380:,380:]=0.2

    monster=np.zeros([11,8])
    monster[0,1]=1
    monster[0,2]=1
    monster[0,3]=1
    monster[1,3]=1
    monster[1,4]=1
    monster[2,1]=1
    monster[2,2]=1
    monster[2,3]=1
    monster[2,4]=1
    monster[2,5]=1
    monster[2,7]=1
    monster[3,0]=1
    monster[3,2]=1
    monster[3,3]=1
    monster[3,5]=1
    monster[3,6]=1
    monster[4,0]=1
    monster[4,2]=1
    monster[4,3]=1
    monster[4,4]=1
    monster[4,5]=1
    monster[5,2]=1
    monster[5,3]=1
    monster[5,4]=1
    monster[5,5]=1
    monster[6,:]=monster[4,:]
    monster[7,:]=monster[3,:]
    monster[8,:]=monster[2,:]
    monster[9,:]=monster[1,:]
    monster[10,:]=monster[0,:]
    monster2=0.2*np.flipud(2*monster.T)

    spship=np.zeros([7,18])
    spship[0,0]=1
    spship[3,0]=1
    spship[6,0]=1
    spship[:,1]=1
    spship[:,2]=1
    spship[1,2]=0
    spship[5,2]=0
    spship[:,3]=1
    spship[1,3]=0
    spship[5,3]=0
    spship[3,4]=1
    spship[3,5]=1
    spship[3,17]=1
    spship[0,15]=1
    spship[6,15]=1

    alphat[Nnodes-10:Nnodes-2,2:13]=1.8*monster2
    alphat[2:10,Nnodes-13:Nnodes-2]=1.8*monster2

    alphat[Nnodes-40:Nnodes-32,42:53]=1.3*monster2
    alphat[42:50,Nnodes-53:Nnodes-42]=1.3*monster2

    alphat=1.1*alphat
    #alphat=(np.diag(np.random.gamma(1.5,0.5,[Nnodes])))
    nout = int(np.floor(Nnodes**2 / 50))
    tmp=np.random.randint(0,Nnodes-1,[2,nout])
    for i in range(nout):
                alphat[tmp[0,i],tmp[1,i]]=alphat[tmp[0,i],tmp[1,i]]+0.2*np.random.gamma(1,0.5)

    alphat[Nnodes // 2 - 3:Nnodes // 2 + 4, 5:23] = spship

    b=0.25*np.ones([Nnodes-50])
    alphat=alphat+np.diag(b,50)+np.diag(b,-50)
    return alphat

with open("src/kafka/enpgf.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile,Loader=yaml.Loader)

Nnodes = cfg["enpgf"]["n"]
dt = cfg["enpgf"]["dt"] 
topic = cfg["enpgf"]["topic"]

betat=7
mut=np.random.gamma(2,0.3,[Nnodes])
Nstep = 10000
alphat = init_alpha(Nnodes)

producer = KafkaProducer(
    bootstrap_servers=['kafka0:29092'],
    value_serializer=lambda x: dumps(x).encode('utf-8')) 

lamb=mut.copy()
for j in range(Nstep):
    print("Kafka producer iteration:", j, "number of nodes:",Nnodes)
    dN_now=np.random.poisson(lamb*dt)
    lamb = mut+(lamb-mut)*(1-betat*dt)+np.dot(alphat,dN_now)
    data = {'step':j,
            'spikes': dN_now.tolist()}
    producer.send(topic, value=data)
    sleep(0.5)
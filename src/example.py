# this script reproduces the example given at the page https://sdelahaies.github.io/enpgf-lab.html

from EnPGF_lab import *

dN,n,nstep = dN_load('../data/dN_100000_128.csv.gzip')
enpgf= EnPGF(n=n,nens=128,dN=dN,dt=0.1,device='cuda')
enpgf.train(nstep = 100000)

enpgf.display_alpha(truth="../data/alphat_128_b.csv")
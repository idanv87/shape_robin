import os
import sys
import math
from matplotlib.ticker import ScalarFormatter

import time

from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import sys
from scipy.interpolate import Rbf

from utils import upsample
from constants import Constants
from utils import  grf

from two_d_data_set import *
from draft import create_data, expand_function
from packages.my_packages import Gauss_zeidel, interpolation_2D

from two_d_model import  deeponet_f2, deeponet
from test_deeponet import domain


def deeponet( F, X, Y):
    
    x_domain=X.flatten()
    y_domain=Y.flatten()
    int_points=np.vstack([x_domain,y_domain]).T
 

    model=deeponet(2,80)
    best_model=torch.load(Constants.path+'runs/'+'2024.04.26.05.19.42best_model.pth')
    model.load_state_dict(best_model['model_state_dict'])
  

    with torch.no_grad():
       
        y1=torch.tensor(int_points,dtype=torch.float32).reshape(int_points.shape)
        f=torch.tensor(F.reshape(1,F.shape[0]),dtype=torch.float32).repeat(y1.shape[0],1)
        dom=torch.tensor([1,1],dtype=torch.float32).repeat(y1.shape[0],1)
        pred2=model([y1, f,dom])

    return pred2
   
n=30
x=np.linspace(0,1,n)
y=np.linspace(0,1,n)
d=domain(x,y,1,1)
xx,yy=np.meshgrid(x,y,indexing='ij')
f=grf(int((n/2))**2, 1,seed=1 )

g1=grf(4*n-4, 1,seed=1)
g2=grf(4*n-4, 1,seed=1 )
g=(g1+Constants.l*g2)[0]

f=upsample(f[0],int(n/2))
f=(f-np.mean(f))/np.std(f)
ga=g[:n]
gb=g[n-1:2*n-1]
gc=g[2*n-2:3*n-2]
gd=np.concatenate([g[3*n-3:4*n-3], [g[0]]])
A,b=d.solver(f.reshape((n,n)),[ga*0,gb*0,gc*0,gd*0])
x0=b*0

def hints(A,b,x0):
    for k in range(1000):
        if k%2==0:
            f=(A@x0-b)
            mu=np.mean(f)
            s=np.std(f)
            corr=(deeponet((f-mu)/s,xx,yy)+mu/s)*s
            x0=x0+corr
            
        else:
            x0=Gauss_zeidel(A.todense(),b,x0)[0]
        print(np.linalg.norm(A@x0-b)/np.linalg.norm(b))
        
hints(A,b,x0)     

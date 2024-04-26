import numpy as np
import scipy
from scipy.linalg import circulant
from scipy.sparse import  kron, identity, csr_matrix
from scipy.stats import qmc
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import torch
from two_d_data_set import *
from two_d_model import  deeponet
from test_deeponet import domain

# from draft import create_data, expand_function
# from geometry import Rect
import time

from utils import count_trainable_params, extract_path_from_dir, save_uniqe, grf, bilinear_upsample,upsample, generate_random_matrix
from constants import Constants
# names=[(1,1), (1,0.9), (1,0.8), (1,0.7), (1,0.6), (1,0.5)]
names=[(1,1)]

def generate_f_g(n, seedf,seedg1, seedg2):

        f=generate_random_matrix(int((n/2))**2,seed=seedf)
        g1=generate_random_matrix(int((4*n-4)/2),seed=seedg1)
        g2=generate_random_matrix(int((4*n-4)/2),seed=seedg2)
        
        f=upsample(f,int(n/2))
        g1=bilinear_upsample(g1)
        g2=bilinear_upsample(g2)
        
        f=(f-np.mean(f))/np.std(f)
        g1=(g1-np.mean(g1))/np.std(g1)
        g2=(g2-np.mean(g2))/np.std(g2)
        
        g=(g1+Constants.l*g2)
        ga=g[:n]
        gb=g[n-1:2*n-1]
        gc=g[2*n-2:3*n-2]
        gd=np.concatenate([g[3*n-3:4*n-3], [g[0]]])
        return f,ga,gb,gc,gd
    
def generate_data(names,  save_path, number_samples,Seed=None):
    X=[]
    Y=[]
    n=30
    x=np.linspace(0,1,n)
    y=np.linspace(0,1,n)
  
    xx,yy=np.meshgrid(x,y,indexing='ij')
    xx=xx.flatten()
    yy=yy.flatten()
    

    for _,dom in enumerate(names):
        d=domain(x,y,dom[0],dom[1])

        for i in range(number_samples):
            try:
                f,ga,gb,gc,gd=generate_f_g(n, Seed,Seed, Seed)
            except:
                f,ga,gb,gc,gd=generate_f_g(n, i,i+1, i+2)

            A,G=d.solver(f.reshape((n,n)),[ga*0,gb*0,gc*0,gd*0])
            # A,G=d.solver(0*upsample(f[0],int(n/2)).reshape((n,n)),[ga,gb,gc,gd])
            u=scipy.sparse.linalg.spsolve(A, G)
            for j in range(len(xx)):
                
             
                X1=[
                    torch.tensor([xx[j],yy[j]], dtype=torch.float32),
                    # torch.tensor(np.concatenate([g.real, g.imag]), dtype=torch.float32),
                    torch.tensor(f, dtype=torch.float32),
                    torch.tensor([dom[0], dom[1]], dtype=torch.float32),
                    # ,
                    ]
                Y1=torch.tensor(u[j], dtype=torch.cfloat)
                save_uniqe([X1,Y1],save_path)
                X.append(X1)
                Y.append(Y1)
               
    return X,Y        

# 

if __name__=='__main__':
    pass
# if False:
    X,Y=generate_data(names, Constants.train_path, number_samples=400, Seed=None)

    X_test, Y_test=generate_data(names,Constants.test_path,number_samples=1, Seed=800)


# fig,ax=plt.subplots()
# for x in X:
#     ax.plot(x[1],'r')
# for x in X_test:
#     ax.plot(x[1],'b')


else:    
    train_data=extract_path_from_dir(Constants.train_path)
    test_data=extract_path_from_dir(Constants.test_path)
    start=time.time()
    s_train=[torch.load(f) for f in train_data]
    print(f"loading torch file take {time.time()-start}")
    s_test=[torch.load(f) for f in test_data]


    X_train=[s[0] for s in s_train]
    Y_train=[s[1] for s in s_train]
    X_test=[s[0] for s in s_test]
    Y_test=[s[1] for s in s_test]







# if __name__=='__main__':
    start=time.time()
    train_dataset = SonarDataset(X_train, Y_train)
    print(f"third loop {time.time()-start}")
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    start=time.time()
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    print(f"4th loop {time.time()-start}")

    train_dataloader = create_loader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False)
    val_dataloader=create_loader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=False)

test_dataset = SonarDataset(X_test, Y_test)
test_dataloader=create_loader(test_dataset, batch_size=Constants.batch_size, shuffle=False, drop_last=False)

inp, out=next(iter(test_dataset))


# model=deeponet_f2(2, 60) 
n=30
model=deeponet(dim=2,f_shape=n**2, domain_shape=2, p=80) 

inp, out=next(iter(test_dataloader))
model(inp)
print(f" num of model parameters: {count_trainable_params(model)}")
# model([X[0].to(Constants.device),X[1].to(Constants.device)])


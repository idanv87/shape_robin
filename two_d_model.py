import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from constants import Constants
import torch.nn.functional as F


import numpy as np
import os
import sys

# 2024.04.24.15.14.27best_model.pth works for f real 1=b=1
 
import torch
import torch.nn as nn

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 80)
        self.fc4 = nn.Linear(80, output_size)  # Assuming you want an output of size 60

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.Tanh()(self.fc2(x))
        x=torch.nn.Tanh()(self.fc3(x))
        x=self.fc4(x)
        return x






class fc(torch.nn.Module):
    def __init__(self, input_shape, output_shape, num_layers, activation_last):
        super().__init__()
        self.activation_last=activation_last
        self.input_shape = input_shape
        self.output_shape = output_shape
        n = 100
        # self.activation = torch.nn.ReLU()
        # self.activation=torch.nn.LeakyReLU()
        self.activation = torch.nn.Tanh()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=self.input_shape, out_features=n, bias=True)])
        output_shape = n

        for j in range(num_layers):
            layer = torch.nn.Linear(
                in_features=output_shape, out_features=n, bias=True)
            # initializer(layer.weight)
            output_shape = n
            self.layers.append(layer)

        self.layers.append(torch.nn.Linear(
            in_features=output_shape, out_features=self.output_shape, bias=True))

    def forward(self, y):
        s=y
        for layer in self.layers:
            s = layer(self.activation(s))
        if self .activation_last:
            return self.activation(s)
        else:
            return s


class deeponet_f(nn.Module):
    def __init__(self, dim,f_shape, domain_shape,  p):
        super().__init__()
        
        n_layers = 4
        self.n = p
        
        self.branch1=FullyConnectedLayer(f_shape,p)
        # self.branch1 = fc(f_shape, p, n_layers,activation_last=False)
        self.branch2=fc(domain_shape, p, n_layers,activation_last=False)
        self.trunk1 = fc(dim, p,  n_layers, activation_last=False)
        # self.bias1 =fc( f_shape+domain_shape+dim, 1, 4, False) 
        self.bias1 =fc( p*3, 1, 4, False)




    def forward(self, X):
        y,f,dom=X

        
        branch1 = self.branch1(f)
        branch2 = self.branch2(dom)
        trunk = self.trunk1(y)
        # bias = torch.squeeze(self.bias1(torch.cat((f,dom,y),dim=1)))
        bias = torch.squeeze(self.bias1(torch.cat((branch1,branch2,trunk),dim=1)))

        return torch.sum(branch1*branch2*trunk, dim=-1, keepdim=False)+bias




    
class deeponet(nn.Module):  
        def __init__(self, dim,f_shape, domain_shape,  p):
            super().__init__()
            self.dim=dim
            self.p=p
            self.model1=deeponet_f(dim,f_shape, domain_shape,  p)
            self.model2=deeponet_f(dim,f_shape, domain_shape,  p)
            
        def forward(self, X):
   
            return self.model1(X)+1J*self.model2(X)

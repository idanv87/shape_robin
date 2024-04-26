
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import multiprocessing
import timeit
import datetime
import os
import time
import torch
from scipy import interpolate
from constants import Constants

data=torch.load(Constants.outputs_path+str('kJM=110_2_50')+'tab1.pt')



fig, ax=plt.subplots(1)
ax.plot(range(len(data['err'])),data['err'])
ax.set_xlabel('iter')
ax.set_ylabel('err')


inset_axes = inset_axes(ax, 
                    width="50%", # width = 30% of parent_bbox
                    height=1.0, # height : 1 inch
                    loc=1)
plt.plot(data['x'],data['u'],'.r',label='numeric')
plt.plot(data['x'],data['solution'],'b',label='analytic')
plt.title('err='+'%.2E' % Decimal(str(data['err'][-1])))
plt.legend()
plt.xticks([])
plt.yticks([])
plt.xlabel('x')



fig.savefig(Constants.eps_fig_path+'fig1_robin.eps', format='eps', bbox_inches='tight')
plt.show()

# plt.show()

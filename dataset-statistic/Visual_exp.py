#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:44:03 2020

@author: zijun.cui
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import pickle
import os
import scipy.io as sio
import numpy as np

pickleFolder = os.path.join( os.getcwd()) 
data = sio.loadmat(os.path.join(pickleFolder, 'BP4D_8AU_6Exp.mat'))
index = data['BP4D_8AU_6Exp']
label_expB = index[0,0]['EXP']

data = sio.loadmat(os.path.join(pickleFolder, 'MMI_8AU_6Exp.mat'))
index = data['MMI_8AU_6Exp']
label_expM = index[0,0]['EXP']

data=sio.loadmat('CK+_6_BASIC.mat')
path=data['path']
label=data['labels']
for i in range(label.shape[0]):
    if label[i,0]==1:
        label[i,0]=label[i,0]+1
label_expC = label[:,0]-1

data = sio.loadmat(os.path.join(pickleFolder, 'EM_11AUs.mat'))
index = data['EM']
sub_name = index[0,0]['SUB_new']
label_expE = index[0,0]['EXP_new']

exp_count = np.zeros([6,4])
for i in np.arange(6):
    exp_count[i,0] = len(np.where(label_expB==(i+1))[0]) #BP4D
    exp_count[i,1] = len(np.where(label_expC==(i+1))[0]) #CK+
    exp_count[i,2] = len(np.where(label_expM==(i+1))[0]) #MMI
    exp_count[i,3] = len(np.where(label_expE==(i+1))[0]) #MMI
    

# Data
r = [0,1,2,3]
ry = [0, 20, 40, 60, 80, 100]
raw_data = {'exp1Bars': exp_count[0,:], 'exp2Bars': exp_count[1,:],\
            'exp3Bars': exp_count[2,:],'exp4Bars': exp_count[3,:],\
            'exp5Bars': exp_count[4,:], 'exp6Bars': exp_count[5,:]}
df = pd.DataFrame(raw_data)
 
# From raw value to percentage
totals = [i+j+k+l+m+n for i,j,k,l,m,n in zip(df['exp1Bars'], df['exp2Bars'],\
                                             df['exp3Bars'],df['exp4Bars'],\
                                             df['exp5Bars'],df['exp6Bars'])]
exp1Bars = [i / j * 100 for i,j in zip(df['exp1Bars'], totals)]
exp2Bars = [i / j * 100 for i,j in zip(df['exp2Bars'], totals)]
exp3Bars = [i / j * 100 for i,j in zip(df['exp3Bars'], totals)]
exp4Bars = [i / j * 100 for i,j in zip(df['exp4Bars'], totals)]
exp5Bars = [i / j * 100 for i,j in zip(df['exp5Bars'], totals)]
exp6Bars = [i / j * 100 for i,j in zip(df['exp6Bars'], totals)]
 
# plot
barWidth = 0.85
names = ('BP4D','CK+','MMI','EmNet')
namesy = ('0','0.2','0.4','0.6','0.8','1')

axis_font = {'fontname':'Arial', 'size':'16'}

#1: Anger 2:Disgust 3:Fear 4: Happy 5: Sad 6: Surprise
# Create green Bars
plt.bar(r, exp1Bars, color='lightcoral', edgecolor='white', width=barWidth, label='Anger')
# Create orange Bars
plt.bar(r, exp2Bars, bottom=exp1Bars, color='peachpuff', edgecolor='white', width=barWidth, label='Disgust')
# Create blue Bars
plt.bar(r, exp3Bars, bottom=[i+j for i,j in zip(exp1Bars, exp2Bars)], \
                             color='lightgreen', edgecolor='white', width=barWidth, label='Fear')

plt.bar(r, exp4Bars, bottom=[i+j+k for i,j,k in zip(exp1Bars, exp2Bars, exp3Bars)], \
                             color='lightskyblue', edgecolor='white', width=barWidth, label='Happy')

plt.bar(r, exp5Bars, bottom=[i+j+k+l for i,j,k,l in zip(exp1Bars, exp2Bars, exp3Bars, exp4Bars)], \
                             color='lavender', edgecolor='white', width=barWidth, label='Sad')

plt.bar(r, exp6Bars, bottom=[i+j+k+l+m for i,j,k,l,m in zip(exp1Bars, exp2Bars, exp3Bars, exp4Bars,exp5Bars)], \
                             color='palevioletred', edgecolor='white', width=barWidth, label='Surprise')


# Add a legend
plt.legend(handletextpad=0.1, loc='upper center', bbox_to_anchor=(0.5,1.26), ncol=3, fancybox=True, prop={'size':14})
#plt.legend(handletextpad=0.1)
# Custom x axis
plt.xticks(r, names, **axis_font)
plt.yticks(ry, namesy, **axis_font)
plt.ylabel('Percentage', **axis_font)
plt.xlabel("Dataset", **axis_font)
plt.savefig('Expression-statistic.pdf', bbox_inches='tight',dpi=300)
# Show graphic
plt.show()
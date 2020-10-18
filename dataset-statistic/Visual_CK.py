#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:33:37 2020

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

data=sio.loadmat('CK+_6_BASIC.mat')
path=data['path']
label=data['labels']
label_exp = label[:,0]
label_AU = label[:,1:]

total_sample = len(path)
AU_count = np.zeros([2,8])
for i in np.arange(8):
    AU_count[0,i] = len(np.where(label_AU[:,i]==0)[0])
    AU_count[1,i] = len(np.where(label_AU[:,i]==1)[0])

# Data
r = [0,1,2,3,4,5,6,7]
ry = [0, 20, 40, 60, 80, 100]
raw_data = {'greenBars': AU_count[0,:], 'blueBars': AU_count[1,:]}
df = pd.DataFrame(raw_data)
 
# From raw value to percentage
totals = [i+j for i,j in zip(df['greenBars'], df['blueBars'])]
greenBars = [i / j * 100 for i,j in zip(df['greenBars'], totals)]
blueBars = [i / j * 100 for i,j in zip(df['blueBars'], totals)]
 
# plot
barWidth = 0.85
names = ('AU1','AU2','AU4','AU6','AU7','AU12','AU15','AU17')
namesy = ('0','0.2','0.4','0.6','0.8','1')
axis_font = {'fontname':'Arial', 'size':'16'}

# Create green Bars
plt.bar(r, greenBars, color='lightgray', edgecolor='white', width=barWidth, label='OFF')
# Create orange Bars
#plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=greenBars, color='greenyellow', edgecolor='white', width=barWidth, label='ON')

# Add a legend
plt.legend(handletextpad=0.1,loc='upper center', bbox_to_anchor=(0.5,1.17), ncol=2, fancybox=True, prop={'size':16})

# Custom x axis
plt.xticks(r, names, **axis_font)
plt.yticks(ry, namesy, **axis_font)
plt.ylabel('Percentage', **axis_font)
plt.xlabel("CK+", **axis_font)
plt.savefig('CK-AU-statistic.pdf', bbox_inches='tight',dpi=300)
# Show graphic
plt.show()
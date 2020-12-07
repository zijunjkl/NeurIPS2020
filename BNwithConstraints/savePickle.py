# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:52:35 2018

@author: Zijun Cui
"""

import pickle
import os
from scipy.io import loadmat
import numpy as np

pickleFolder = os.path.join( os.getcwd()) 
# bp4d task 
#task 1: happy; task 2: sad; task 3: surprise; Task 5: fear; Task 7: Anger; Task 8: Disgust
# expression index from constraints: 1: Happy 2:Sad 3:Surprise 4:Fear 5: Anger 6: Disgust
# expression index for CK+ in NN: 1: Anger 2:Disgust 3:Fear 4: Happy 5: Sad 6: Surprise

#prob_arr = loadmat('./ExpDepSingle.mat')['prob_arr']
#config_list = loadmat('./ExpDepSingle.mat')['config_list']

#prob_arr = loadmat('./ExpDepJoint.mat')['prob_arr']
#config_list = loadmat('./ExpDepJoint.mat')['config_list']

#prob_arr = loadmat('./ExpIndepSingle.mat')['prob_arr']
#config_list = loadmat('./ExpDepSingle.mat')['config_list']

prob_arr = loadmat('./Total.mat')['prob_arr']
config_list = loadmat('./Total.mat')['config_list']

list_AU_configs = loadmat('list_AU_config')['list_AU_config']

exp_list = np.array([5, 6, 4, 1, 2, 3])
PGM_p_AUconfig = np.zeros([6,256])
for i in np.arange(6):
    exp = exp_list[i]-1
    for AU_config in np.arange(256):
        arr = list_AU_configs[AU_config,:]
        arr = np.insert(arr, 0, exp)
        idx = (config_list == arr).all(axis=1).nonzero()[0]
        PGM_p_AUconfig[i, AU_config] = prob_arr[0,idx]
    
    PGM_p_AUconfig[i,:] = PGM_p_AUconfig[i,:]/sum(PGM_p_AUconfig[i,:])

PGM_p_Exp = np.zeros([256,6])
for AU_config in np.arange(256):
    for i in np.arange(6):
        arr = list_AU_configs[AU_config,:]
        exp = exp_list[i]-1
        arr = np.insert(arr, 0, exp)
        idx = (config_list == arr).all(axis=1).nonzero()[0]
        PGM_p_Exp[AU_config, i] = prob_arr[0,idx]
    
    PGM_p_Exp[AU_config,:] = PGM_p_Exp[AU_config,:]/sum(PGM_p_Exp[AU_config,:])
    
    

pickleFolder = os.path.join( os.getcwd(), 'PseudoBN') 

#pickle_out = open( os.path.join( pickleFolder,'ExpDepSingle', 'PGM_p.p'), "wb" )
#pickle.dump([PGM_p_Exp, PGM_p_AUconfig], pickle_out)

#pickle_out = open( os.path.join( pickleFolder, 'ExpDepJoint', 'PGM_p.p'), "wb" )
#pickle.dump([PGM_p_Exp, PGM_p_AUconfig], pickle_out)

#pickle_out = open( os.path.join( pickleFolder, 'ExpIndepJoint', 'PGM_p.p'), "wb" )
#pickle.dump([PGM_p_Exp, PGM_p_AUconfig], pickle_out)

pickle_out = open( os.path.join( pickleFolder, 'PseudoBN', 'Total', 'PGM_p.p'), "wb" )
pickle.dump([PGM_p_Exp, PGM_p_AUconfig], pickle_out)


pickle_out.close()



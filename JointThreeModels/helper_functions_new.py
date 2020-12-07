# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:29:03 2019

@author: Zijun Cui
"""

import numpy as np
import tensorflow as tf
import matplotlib.image as mping
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import f1_score
from scipy.io import loadmat
from os.path import isfile, join
import skimage.io
import pdb
def get_train_batch(batch_size, Data_index, Label, PGM_p_AUconfig, start_idx):
    
    num_sample = len(Data_index)
    batch_end = min(start_idx + batch_size, num_sample)
    batch_D_index = Data_index[start_idx : batch_end]
    #batch_D = []
    # expression labels: (0,1,2,3,4) + 5 for neutral
    batch_exp_label= Label[start_idx : batch_end, 0] - 1 # expression labels 
    #batch_AU_label = Label[start_idx : batch_end, 1 : Label.shape[1]] # AU labels
    batch_AU_prob  = []
    
    for i in range(len(batch_D_index)):
        image_path = batch_D_index[i]
    #for image_path in batch_D_index:
        img = skimage.io.imread(image_path)
        if img.shape[2] == 3:
            img = img/255.0
            #re_img = skimage.transform.resize(img, (32, 32, 3))
            if i==0:
                batch_D=np.expand_dims(img, axis=0)
            else:
                batch_D=np.concatenate((batch_D, np.expand_dims(img, axis=0)), axis=0)
            #batch_D.append(re_img/np.max(re_img))
        exp = batch_exp_label[i]
        #batch_AU_prob.append(PGM_p_AUconfig[exp[0,0], :])
#        if np.sum(batch_AU_label[i,:]) == 0:
#            batch_exp_label[i] = [5]


    
    batch_exp_label = np.ravel(batch_exp_label)
    return batch_D, batch_exp_label#, batch_AU_label#, batch_AU_prob
def get_train_batch_ck(batch_size, Data_index, Label, PGM_p_AUconfig, start_idx):
    
    num_sample = len(Data_index)
    batch_end = min(start_idx + batch_size, num_sample)
    batch_D_index = Data_index[start_idx : batch_end]
    #batch_D = []
    # expression labels: (0,1,2,3,4) + 5 for neutral
    batch_exp_label= Label[start_idx : batch_end, 0] - 1 # expression labels 
    #batch_AU_label = Label[start_idx : batch_end, 1 : Label.shape[1]] # AU labels
    batch_AU_prob  = []
    
    for i in range(len(batch_D_index)):
        image_path = batch_D_index[i]
    #for image_path in batch_D_index:
        img = skimage.io.imread(image_path)
        if img.shape[2] == 3:
            img = img/255.0
            #re_img = skimage.transform.resize(img, (32, 32, 3))
            if i==0:
                batch_D=np.expand_dims(img, axis=0)
            else:
                batch_D=np.concatenate((batch_D, np.expand_dims(img, axis=0)), axis=0)
            #batch_D.append(re_img/np.max(re_img))
        exp = batch_exp_label[i]
        #batch_AU_prob.append(PGM_p_AUconfig[exp[0,0], :])
#        if np.sum(batch_AU_label[i,:]) == 0:
#            batch_exp_label[i] = [5]


    
    batch_exp_label = np.ravel(batch_exp_label)
    return batch_D, batch_exp_label#, batch_AU_label#, batch_AU_prob
def get_train_batch_AU(batch_size, Data_index, Label, start_idx):
    
    num_sample = len(Data_index)
    batch_end = min(start_idx + batch_size, num_sample)
    batch_D_index = Data_index[start_idx : batch_end]
    #batch_D = []
    # expression labels: (0,1,2,3,4) + 5 for neutral
    #batch_exp_label= Label[start_idx : batch_end, 0] - 1 # expression labels
    batch_AU_label = Label[start_idx : batch_end, : ] # AU labels
    #batch_AU_label_out = np.concatenate([batch_AU_label, np.expand_dims(Label[start_idx : batch_end, 5] ,axis=1),np.expand_dims(Label[start_idx : batch_end, 6] ,axis=1),np.expand_dims(Label[start_idx : batch_end, 9] ,axis=1)],axis=1)
    
    batch_AU_prob  = []
    for i in range(len(batch_D_index)):
        image_path = batch_D_index[i]
    #for image_path in batch_D_index:
        img = skimage.io.imread(image_path)
        if img.shape[2] == 3:
            img = img/255.0
            #re_img = skimage.transform.resize(img, (32, 32, 3))
            if i==0:
                batch_D=np.expand_dims(img, axis=0)
            else:
                batch_D=np.concatenate((batch_D, np.expand_dims(img, axis=0)), axis=0)
            #batch_D.append(re_img/np.max(re_img))
        #exp = batch_exp_label[i]
        #batch_AU_prob.append(PGM_p_AUconfig[exp[0,0], :])
        #        if np.sum(batch_AU_label[i,:]) == 0:
        #            batch_exp_label[i] = [5]


    
    #batch_exp_label = np.ravel(batch_exp_label)
    return batch_D,  batch_AU_label#, batch_AU_prob

def get_valid_test_set(Data_index, Label, PGM_p_AUconfig):

    #get_data = []
    AU_prob = []
    exp_label= Label[:, 0] - 1 # expression labels
    AU_label = Label[:, 1:Label.shape[1]] # AU labels
    for i in range(len(Data_index)):
        image_path = Data_index[i]
    #for image_path in Data_index:
        img = skimage.io.imread(image_path)
        if img.shape[2] == 3:
            img = img/255.0
            #re_img = skimage.transform.resize(img, (32, 32, 3))
            if i==0:
                get_data=np.expand_dims(img, axis=0)
            else:
                get_data=np.concatenate((get_data, np.expand_dims(img, axis=0)), axis=0)
            #get_data.append(re_img/np.max(re_img))
        exp = exp_label[i]
        AU_prob.append(PGM_p_AUconfig[exp[0,0], :])
#        if np.sum(AU_label[i,:]) == 0:
#            exp_label[i] = [5]
#    for exp in exp_label:
#        AU_prob.append(PGM_p_AUconfig[exp[0,0]-1, :])
    
    exp_label= np.ravel(exp_label) # expression labels
    return get_data, exp_label, AU_label, AU_prob

def Compute_Accuracy(label, pred):
    right = 0
    right_ = np.zeros((10))
    count = np.zeros((10))
    for i in range(len(label)):
        class_idx = label[i] 
        count[class_idx] = count[class_idx] + 1
        if pred[i] == label[i]:
            right += 1
            right_[class_idx] = right_[class_idx] + 1
    for i in range(10):
        right_[i] = right_[i]/count[i]
    return right/len(label), right_

def Compute_F1score(label, pred):
    return f1_score(label, pred, average='binary') #expression starting from 0


def Compute_F1score_au(label, pred):
    num_AU = label.shape[1]
    f1_score_eachAU = np.zeros((1, num_AU))
    for i in range(num_AU): # number of AUs
        f1_score_eachAU[0,i] = Compute_F1score(label[:,i], pred[:,i])
    return f1_score_eachAU

def weight_variable(name, shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
#    return tf.Variable(tf.random.normal(shape, dtype = tf.float32, stddev = 0.35), name = name)


def bias_variable(name, shape):
    return tf.Variable(initial_value=tf.constant(0.1, shape=shape), name = name)
    #return tf.Variable(tf.zeros(shape, dtype = tf.float32), name = name)


def conv2d(x, W, b):
    x_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #return output of activation layer
    return tf.nn.relu(x_conv + b)
    #return x

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


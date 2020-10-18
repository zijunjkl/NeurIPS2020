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
from scipy.stats import entropy
import pdb
import cv2

def get_train_batch(batch_size, Data_index, Label, PGM_p_AUconfig, start_idx):
    
    num_sample = len(Data_index)
    batch_end = min(start_idx + batch_size, num_sample)
    batch_D_index = Data_index[start_idx : batch_end]
    #batch_D = []
    # expression labels: (0,1,2,3,4) + 5 for neutral
    batch_exp_label= Label[start_idx : batch_end, 0] - 1 # expression labels 
    batch_AU_label = Label[start_idx : batch_end, 1 : Label.shape[1]] # AU labels
    batch_AU_prob  = []
   
    for i in range(len(batch_D_index)):
        image_path = batch_D_index[i]
    #for image_path in batch_D_index:
        img = cv2.imread(image_path)
        img_224= cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
        if img.shape[2] == 3:
            img = img/255.0
            img_224=img_224/255.0            
            #re_img = skimage.transform.resize(img, (32, 32, 3))
            if i==0:
                batch_D=np.expand_dims(img, axis=0)
                batch_D_224=np.expand_dims(img_224,axis=0)
            else:
                batch_D=np.concatenate((batch_D, np.expand_dims(img, axis=0)), axis=0)
                batch_D_224=np.concatenate((batch_D_224,np.expand_dims(img_224,axis=0)),axis=0)
            #batch_D.append(re_img/np.max(re_img))
        exp = batch_exp_label[i]
        batch_AU_prob.append(PGM_p_AUconfig[exp, :])
    return batch_D, batch_exp_label, batch_AU_label, batch_AU_prob, batch_D_224
def get_train_batch_ck(batch_size, Data_index, Label, PGM_p_AUconfig, start_idx):
    
    num_sample = len(Data_index)
    batch_end = min(start_idx + batch_size, num_sample)
    batch_D_index = Data_index[start_idx : batch_end]
    #batch_D = []
    # expression labels: (1,2,4,5,6,7) for six basic emotions
    
       
    batch_exp_label= Label[start_idx : batch_end, 0] - 2 # expression labels 
    batch_AU_label = Label[start_idx : batch_end, 1 : Label.shape[1]] # AU labels
    batch_AU_prob  = []
    
    for i in range(len(batch_D_index)):
        image_path = batch_D_index[i][1:31]
        #for image_path in batch_D_index:
        img_s = np.expand_dims(cv2.imread('../CK+'+image_path,0),axis=2)
        img = cv2.resize(np.concatenate([img_s,img_s,img_s],axis=2),(64,64),interpolation=cv2.INTER_AREA)
        img_224=cv2.resize(np.concatenate([img_s,img_s,img_s],axis=2),(224,224),interpolation=cv2.INTER_AREA)
        if img.shape[2] == 3:
            img = img/255.0
            img_224 = img_224/225.0
            #re_img = skimage.transform.resize(img, (32, 32, 3))
            if i==0:
                batch_D=np.expand_dims(img, axis=0)
                batch_D_224=np.expand_dims(img_224, axis=0)
            else:
                batch_D=np.concatenate((batch_D, np.expand_dims(img, axis=0)), axis=0)
                batch_D_224=np.concatenate((batch_D_224, np.expand_dims(img_224, axis=0)), axis=0)
                
            #batch_D.append(re_img/np.max(re_img))
        exp = batch_exp_label[i]
        batch_AU_prob.append(PGM_p_AUconfig[exp, :])
    return batch_D, batch_exp_label, batch_AU_label, batch_AU_prob,batch_D_224
def get_train_batch_MMI(batch_size, Data_index, Label, PGM_p_AUconfig, start_idx):
    
    num_sample = len(Data_index)
    batch_end = min(start_idx + batch_size, num_sample)
    batch_D_index = Data_index[start_idx : batch_end]
    #batch_D = []
    # expression labels: (1,2,4,5,6,7) for six basic emotions
    
       
    batch_exp_label= Label[start_idx : batch_end, 0] - 2 # expression labels 
    batch_AU_label = Label[start_idx : batch_end, 1 : Label.shape[1]] # AU labels
    batch_AU_prob  = []
    
    for i in range(len(batch_D_index)):
        image_path = batch_D_index[i]
        #for image_path in batch_D_index:
        img =  cv2.imread('/home/zijun/Documents/MMI/crop_new/'+image_path+'.jpg',cv2.IMREAD_UNCHANGED)
        
        if img.shape[2] == 3:
            img = img/255.0
            #re_img = skimage.transform.resize(img, (32, 32, 3))
            if i==0:
                batch_D=np.expand_dims(img, axis=0)
            else:
                batch_D=np.concatenate((batch_D, np.expand_dims(img, axis=0)), axis=0)
            #batch_D.append(re_img/np.max(re_img))
        exp = batch_exp_label[i]
        batch_AU_prob.append(PGM_p_AUconfig[exp, :])
    
    return batch_D, batch_exp_label, batch_AU_label, batch_AU_prob
def get_train_batch_GT_ck(batch_size, Data_index, Label, start_idx):
    
    num_sample = len(Data_index)
    batch_end = min(start_idx + batch_size, num_sample)
    batch_D_index = Data_index[start_idx : batch_end]
    #batch_D = []
    # expression labels: (1,2,4,5,6,7) for six basic emotions
    
       
    batch_exp_label= Label[start_idx : batch_end, 0] - 2 # expression labels 
    batch_AU_label = Label[start_idx : batch_end, 1 : Label.shape[1]] # AU labels
    
    for i in range(len(batch_D_index)):
        image_path = batch_D_index[i][1:31]
        #for image_path in batch_D_index:
        img_s = np.expand_dims(cv2.imread('../CK+'+image_path,0),axis=2)
        img = cv2.resize(np.concatenate([img_s,img_s,img_s],axis=2),(64,64),interpolation=cv2.INTER_AREA)
        if img.shape[2] == 3:
            img = img/255.0
            #re_img = skimage.transform.resize(img, (32, 32, 3))
            if i==0:
                batch_D=np.expand_dims(img, axis=0)
            else:
                batch_D=np.concatenate((batch_D, np.expand_dims(img, axis=0)), axis=0)
            #batch_D.append(re_img/np.max(re_img))
    return batch_D, batch_exp_label, batch_AU_label
def get_train_batch_pro(batch_size, Data_index, Label, start_idx):
    
    num_sample = len(Data_index)
    batch_end = min(start_idx + batch_size, num_sample)
    batch_D_index = Data_index[start_idx : batch_end]
    #batch_D = []
    # expression labels: (0,1,2,3,4) + 5 for neutral 
    batch_AU_label = Label[start_idx : batch_end, :] # AU labels
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
        #batch_AU_prob.append(PGM_p_AUconfig[exp, :])
#        if np.sum(batch_AU_label[i,:]) == 0:
#            batch_exp_label[i] = [5]


    batch_exp_label=0
    batch_exp_label = np.ravel(batch_exp_label)
    return batch_D, batch_exp_label, batch_AU_label#, batch_AU_prob

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
        AU_prob.append(PGM_p_AUconfig[exp, :])
#        if np.sum(AU_label[i,:]) == 0:
#            exp_label[i] = [5]
#    for exp in exp_label:
#        AU_prob.append(PGM_p_AUconfig[exp[0,0]-1, :])
    
    exp_label= np.ravel(exp_label) # expression labels
    return get_data, exp_label, AU_label, AU_prob

def posterior_summation(p_exp_1, p_exp_2):
    p_exp_sum = p_exp_1 + p_exp_2
#    num_exp = np.shape(p_exp_sum)[1]
#    p_exp_norm = np.tile(np.sum(p_exp_sum, axis=1), [1, num_exp])
    p_exp_norm = 2*np.ones(np.shape(p_exp_sum))
    p_exp = np.divide(p_exp_sum, p_exp_norm)
    
    return p_exp

def posterior_entropy(p_exp_1, p_exp_2):
    p_exp = np.zeros(np.shape(p_exp_1))
    for i in range(np.shape(p_exp_1)[0]):
        s1 = entropy(p_exp_1[i,:])
        s2 = entropy(p_exp_2[i,:])
        if (s1+s2) != 0 :
            w1 = s2/(s1+s2)
            w2 = s1/(s1+s2)
            p = w1*p_exp_1[i,:] + w2*p_exp_2[i,:]
            p = p/sum(p)
            p_exp[i,:] = p
        else:
            p = p_exp_1[i,:] + p_exp_2[i,:]
            p_exp[i,:] = p/sum(p)
            
    return p_exp

def posterior_entropy_three(p_exp_1, p_exp_2, p_exp_3):
    p_exp = np.zeros(np.shape(p_exp_1))
    for i in range(np.shape(p_exp_1)[0]):
        s1 = entropy(p_exp_1[i,:])
        s2 = entropy(p_exp_2[i,:])
        s3 = entropy(p_exp_3[i,:])
        if s1 != 0 :
             w1 = 1/s1
        else:
             w1 = 0
            
        if s2 != 0 :
             w2 = 1/s2
        else:
             w2 = 0   
            
        if s3 != 0 :
             w3 = 1/s3
        else:
             w3 = 0 
        
        if (w1+w2+w3) != 0:
             p = w1*p_exp_1[i,:] + w2*p_exp_2[i,:] + w3*p_exp_3[i,:]
             p_exp[i,:] = p/sum(p)
        else:
             p = p_exp_1[i,:] + p_exp_2[i,:] + p_exp_3[i,:]
             p_exp[i,:] = p/sum(p)
#        p = 0.2*p_exp_1[i,:] + 0.2*p_exp_2[i,:] + 0.6*p_exp_3[i,:]
        p_exp[i,:] = p/sum(p)
    return p_exp
def posterior_entropy_three_2(p_exp_1, p_exp_3):
    p_exp = np.zeros(np.shape(p_exp_1))
    for i in range(np.shape(p_exp_1)[0]):
        s1 = entropy(p_exp_1[i,:])
        #s2 = entropy(p_exp_2[i,:])
        s3 = entropy(p_exp_3[i,:])
        p = 0.3*p_exp_1[i,:]   + 0.7*p_exp_3[i,:]
        p_exp[i,:] = p/sum(p)
    return p_exp
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


# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:32:30 2019

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
from helper_functions import weight_variable, bias_variable, conv2d, max_pool_2x2
import pdb
def Prediction_3rdModel(p_AUs):
    '''
    third model 
    left expression model
    Input p({Zi}|X) 
    Output p(Y|input) 
    '''
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1_3rd", [5, 128]) 
    b_fc1 = bias_variable("b_fc1_3rd", [128])
    #reshape conv2 output to fit fully connected layer input
    
#    num_AUs = tf.shape(p_AUs)[1]
#    num_class = tf.shape(p_AUs)[2]
#    bSize = tf.shape(p_AUs)[0]
#    p_AUs_reshape = tf.reshape(p_AUs, [bSize, num_AUs*num_class])
    h_fc1 = tf.nn.relu(tf.matmul(p_AUs, W_fc1) + b_fc1)
    
    #
    W_fc2 = weight_variable("W_fc2_3rd", [128, 512]) 
    b_fc2 = bias_variable("b_fc2_3rd", [512])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
#    
    # 3 layer
    W_fc3 = weight_variable("W_fc3_3rd", [512, 5]) 
    b_fc3 = bias_variable("b_fc3_3rd", [5])
    #reshape conv2 output to fit fully connected layer input
    h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3
    
    p_Exp = tf.nn.softmax(h_fc3)

    return p_Exp, h_fc3



def Prediction_3rdModel_joint(p_AUs):
    '''
    third model 
    left expression model
    Input p({Zi}|X) 
    Output p(Y|input) 
    '''
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1_3rd", [256, 128]) 
    b_fc1 = bias_variable("b_fc1_3rd", [128])
    #reshape conv2 output to fit fully connected layer input
    
#    num_AUs = tf.shape(p_AUs)[1]
#    num_class = tf.shape(p_AUs)[2]
#    bSize = tf.shape(p_AUs)[0]
#    p_AUs_reshape = tf.reshape(p_AUs, [bSize, num_AUs*num_class])
    h_fc1 = tf.nn.relu(tf.matmul(p_AUs, W_fc1) + b_fc1)
    
    #
    W_fc2 = weight_variable("W_fc2_3rd", [128, 512]) 
    b_fc2 = bias_variable("b_fc2_3rd", [512])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
#    
    # 3 layer
    W_fc3 = weight_variable("W_fc3_3rd", [512, 5]) 
    b_fc3 = bias_variable("b_fc3_3rd", [5])
    #reshape conv2 output to fit fully connected layer input
    h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3
    
    p_Exp = tf.nn.softmax(h_fc3)

    return p_Exp, h_fc3
def Prediction_3rdModel_joint_CK(p_AUs,drop_prob):
    '''
    third model 
    left expression model
    Input p({Zi}|X) 
    Output p(Y|input) 
    '''
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1_3rd", [256, 64]) 
    b_fc1 = bias_variable("b_fc1_3rd", [64])
    #reshape conv2 output to fit fully connected layer input
    
#    num_AUs = tf.shape(p_AUs)[1]
#    num_class = tf.shape(p_AUs)[2]
#    bSize = tf.shape(p_AUs)[0]
#    p_AUs_reshape = tf.reshape(p_AUs, [bSize, num_AUs*num_class])
    h_fc1 = tf.nn.relu(tf.matmul(p_AUs, W_fc1) + b_fc1)
    h_fc1 = tf.nn.dropout(h_fc1,drop_prob)
    #
    W_fc2 = weight_variable("W_fc2_3rd", [64, 64]) 
    b_fc2 = bias_variable("b_fc2_3rd", [64])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2 = tf.nn.dropout(h_fc2,drop_prob)
    # # 3 layer
    W_fc3 = weight_variable("W_fc3_3rd", [64, 64]) 
    b_fc3 = bias_variable("b_fc3_3rd", [64])
    # #reshape conv2 output to fit fully connected layer input
    h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3
    h_fc3 = tf.nn.dropout(h_fc3,drop_prob)

    # 4 layer
    W_fc4 = weight_variable("W_fc4_3rd", [64, 5]) 
    b_fc4 = bias_variable("b_fc4_3rd", [5])
    #reshape conv2 output to fit fully connected layer input
    h_fc4 = tf.matmul(h_fc3, W_fc4) + b_fc4

    p_Exp = tf.nn.softmax(h_fc4)

    return p_Exp, h_fc4

def Loss_3rdModel(label_Expression, p_Exp_3rd):
    #batch_size = tf.shape(pred_exp)[0]
    #loss_exp_gt = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label_exp, logits = pred_exp)) 
    
    depth = tf.shape(p_Exp_3rd)[1]
    one_hot_label_exp = tf.one_hot(label_Expression, depth)
#    loss_exp_gt = tf.reduce_mean(tf.square(pred_exp - one_hot_label_exp))
    loss_exp_gt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_hot_label_exp, logits = p_Exp_3rd))
    
    
    return loss_exp_gt
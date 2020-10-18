# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:32:30 2019

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

def CNN_Expression(x_image,keep_prob):
    '''
    five layer CNN for expression classification
    input should be whole images
    output expression distribution for images
    '''
    # first convolutional layer(+ activation layer)
    W_conv1 = weight_variable("W_conv1", [5, 5, 3, 32])
    b_conv1 = bias_variable("b_conv1", [32])
    h_conv1 = conv2d(x_image, W_conv1, b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #second convolutional layer(+ activation layer)
    
    W_conv2 = weight_variable("W_conv2", [5, 5, 32, 32])
    b_conv2 = bias_variable("b_conv2", [32])
    h_conv2 = conv2d(h_pool1, W_conv2, b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) 
    
    #third convolutional layer(+ activation layer)
    W_conv3 = weight_variable("W_conv3", [3, 3, 32, 32])
    b_conv3 = bias_variable("b_conv3", [32])
    h_conv3 = conv2d(h_pool2, W_conv3, b_conv3) 
    h_pool3 = max_pool_2x2(h_conv3)
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1", [8*8*32, 128]) 
    b_fc1 = bias_variable("b_fc1", [128])
    #reshape conv2 output to fit fully connected layer input
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*32]) 
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  
    h_fc1 = tf.nn.dropout(h_fc1,keep_prob)
    
    #output layer
    W_fc2 = weight_variable("W_fc2", [128, 5]) 
    b_fc2 = bias_variable("b_fc2", [5])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    p_Exp = tf.nn.softmax(h_fc2)
    regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)

    return p_Exp, h_fc2, regularizers

def Loss_ExpressionModel(pred_exp, label_exp, posterior_exp, coeff):
    #batch_size = tf.shape(pred_exp)[0]
    #loss_exp_gt = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label_exp, logits = pred_exp)) 
    
    depth = tf.shape(pred_exp)[1]
    one_hot_label_exp = tf.one_hot(label_exp, depth)
    loss_exp_gt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_hot_label_exp, logits = pred_exp))
    #EXPconfig_tile = tf.tile(EXPconfig, batch_size)
    loss_exp_posterior = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = posterior_exp, logits = pred_exp))

    loss_exp = loss_exp_gt + coeff*loss_exp_posterior
    
    return loss_exp

def Loss_ExpressionModel_Labelonly(pred_exp, label_exp):
    #batch_size = tf.shape(pred_exp)[0]
    #loss_exp_gt = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label_exp, logits = pred_exp)) 
    
    depth = tf.shape(pred_exp)[1]
    one_hot_label_exp = tf.one_hot(label_exp, depth)
#    loss_exp_gt = tf.reduce_mean(tf.square(pred_exp - one_hot_label_exp))
    loss_exp_gt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_hot_label_exp, logits = pred_exp))
    
    
    return loss_exp_gt

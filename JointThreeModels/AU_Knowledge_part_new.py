# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:29:03 2019

@author: Zijun Cui
"""
import tensorflow as tf
from helper_functions import weight_variable, bias_variable, conv2d, max_pool_2x2
import pdb

def CNN_AUdetection(x_image,keep_prob,train_mode):
    '''
    three layer CNN for AU detection
    input should be one patch of images containing certain AUs
    output detection result for one AU
    '''
    
    # first convolutional layer(+ activation layer)
    W_conv1 = weight_variable("W_conv1", [5, 5, 3, 32])
    b_conv1 = bias_variable("b_conv1", [32])
    h_conv1 = conv2d(x_image, W_conv1, b_conv1)#(, 64, 64, 32)
    #h_conv1 = batch_normalization(h_conv1,train_mode,'con1')
    h_pool1 = max_pool_2x2(h_conv1) #(None, 32, 32, 32)
    
    #second convolutional layer(+ activation layer)
    W_conv2 = weight_variable("W_conv2", [5, 5, 32, 64])
    b_conv2 = bias_variable("b_conv2", [64])
    h_conv2 = conv2d(h_pool1, W_conv2, b_conv2)#(None, 32, 32, 64)
    #h_conv2 = batch_normalization(h_conv2,train_mode,'con2')
    h_pool2 = max_pool_2x2(h_conv2) #(None, 16, 16, 64)
    
    #third convolutional layer(+ activation layer)
    W_conv3 = weight_variable("W_conv3", [3, 3, 64, 64])
    b_conv3 = bias_variable("b_conv3", [64])
    h_conv3 = conv2d(h_pool2, W_conv3, b_conv3) #(None, 16, 16, 64)
    #h_conv3 = batch_normalization(h_conv3,train_mode,'con3')
    h_pool3 = max_pool_2x2(h_conv3) #(None, 8, 8, 64)
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1", [8*8*64, 512]) 
    b_fc1 = bias_variable("b_fc1", [512])
    #reshape conv2 output to fit fully connected layer input
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*64]) 
    h_pool3_flat = tf.nn.dropout(h_pool3_flat,keep_prob)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  
    h_fc1 = tf.nn.dropout(h_fc1,keep_prob)
    
    #output layer
    W_fc2 = weight_variable("W_fc2", [512, 22]) 
    b_fc2 = bias_variable("b_fc2", [22])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    p_AU1 = tf.nn.softmax(h_fc2[:,0:2])
    p_AU2 = tf.nn.softmax(h_fc2[:,2:4])
    p_AU3 = tf.nn.softmax(h_fc2[:,4:6])
    p_AU4 = tf.nn.softmax(h_fc2[:,6:8])
    p_AU5 = tf.nn.softmax(h_fc2[:,8:10])
    p_AU6 = tf.nn.softmax(h_fc2[:,10:12])
    p_AU7 = tf.nn.softmax(h_fc2[:,12:14])
    p_AU8 = tf.nn.softmax(h_fc2[:,14:16])
    p_AU9 = tf.nn.softmax(h_fc2[:,16:18])
    p_AU10 = tf.nn.softmax(h_fc2[:,18:20])
    p_AU11 = tf.nn.softmax(h_fc2[:,20:22])
    return p_AU1,p_AU2,p_AU3,p_AU4, p_AU5, p_AU6,p_AU7,p_AU8,p_AU9,p_AU10,p_AU11,h_fc2

def CNN_AUdetection_ck(x_image,keep_prob,train_mode):
    '''
    three layer CNN for AU detection
    input should be one patch of images containing certain AUs
    output detection result for one AU
    '''
    
    # first convolutional layer(+ activation layer)
    W_conv1 = weight_variable("W_conv1", [5, 5, 3, 64])
    b_conv1 = bias_variable("b_conv1", [64])
    h_conv1 = conv2d(x_image, W_conv1, b_conv1)#(, 64, 64, 32)
    #h_conv1 = batch_normalization(h_conv1,train_mode,'con1')
    h_pool1 = max_pool_2x2(h_conv1) #(None, 32, 32, 32)
    
    #second convolutional layer(+ activation layer)
    W_conv2 = weight_variable("W_conv2", [5, 5, 64, 64])
    b_conv2 = bias_variable("b_conv2", [64])
    h_conv2 = conv2d(h_pool1, W_conv2, b_conv2)#(None, 32, 32, 64)
    #h_conv2 = batch_normalization(h_conv2,train_mode,'con2')
    h_pool2 = max_pool_2x2(h_conv2) #(None, 16, 16, 64)
    
    #third convolutional layer(+ activation layer)
    W_conv3 = weight_variable("W_conv3", [3, 3, 64, 64])
    b_conv3 = bias_variable("b_conv3", [64])
    h_conv3 = conv2d(h_pool2, W_conv3, b_conv3) #(None, 16, 16, 64)
    #h_conv3 = batch_normalization(h_conv3,train_mode,'con3')
    h_pool3 = max_pool_2x2(h_conv3) #(None, 8, 8, 64)
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1", [8*8*64, 1024]) 
    b_fc1 = bias_variable("b_fc1", [1024])
    #reshape conv2 output to fit fully connected layer input
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*64]) 
    h_pool3_flat = tf.nn.dropout(h_pool3_flat,keep_prob)
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  
    h_fc1 = tf.nn.dropout(h_fc1,keep_prob)
    
    #output layer
    W_fc2 = weight_variable("W_fc2", [1024, 16]) 
    b_fc2 = bias_variable("b_fc2", [16])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    p_AU1 = tf.nn.softmax(h_fc2[:,0:2])
    p_AU2 = tf.nn.softmax(h_fc2[:,2:4])
    p_AU3 = tf.nn.softmax(h_fc2[:,4:6])
    p_AU4 = tf.nn.softmax(h_fc2[:,6:8])
    p_AU5 = tf.nn.softmax(h_fc2[:,8:10])
    p_AU6 = tf.nn.softmax(h_fc2[:,10:12])
    p_AU7 = tf.nn.softmax(h_fc2[:,12:14])
    p_AU8 = tf.nn.softmax(h_fc2[:,14:16])
    return p_AU1,p_AU2,p_AU3,p_AU4, p_AU5, p_AU6,p_AU7,p_AU8,h_fc2


def CNN_AUdetection_joint(x_image,keep_prob):
    '''
    three layer CNN for AU detection
    input should be one patch of images containing certain AUs
    output detection result for joint AU
    '''
    
    # first convolutional layer(+ activation layer)
    W_conv1 = weight_variable("W_conv1", [5, 5, 3, 32])
    b_conv1 = bias_variable("b_conv1", [32])
    h_conv1 = conv2d(x_image, W_conv1, b_conv1)#(, 64, 64, 32)
    h_pool1 = max_pool_2x2(h_conv1) #(None, 32, 32, 32)
    
    #second convolutional layer(+ activation layer)
    W_conv2 = weight_variable("W_conv2", [5, 5, 32, 64])
    b_conv2 = bias_variable("b_conv2", [64])
    h_conv2 = conv2d(h_pool1, W_conv2, b_conv2)#(None, 32, 32, 64)
    h_pool2 = max_pool_2x2(h_conv2) #(None, 16, 16, 64)
    
    #third convolutional layer(+ activation layer)
    W_conv3 = weight_variable("W_conv3", [3, 3, 64, 256])
    b_conv3 = bias_variable("b_conv3", [256])
    h_conv3 = conv2d(h_pool2, W_conv3, b_conv3) #(None, 16, 16, 64)
    h_pool3 = max_pool_2x2(h_conv3) #(None, 8, 8, 64)
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1", [8*8*256, 8192]) 
    b_fc1 = bias_variable("b_fc1", [8192])
    #reshape conv2 output to fit fully connected layer input
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*256]) 
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  
    h_fc1 = tf.nn.dropout(h_fc1,keep_prob)
    
    #output layer
    W_fc2 = weight_variable("W_fc2", [8192, 2048]) 
    b_fc2 = bias_variable("b_fc2", [2048])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    p_AUs = tf.nn.softmax(h_fc2)
    
    return p_AUs, h_fc2

def CNN_AUdetection_joint_ck(x_image,keep_prob):
    '''
    three layer CNN for AU detection
    input should be one patch of images containing certain AUs
    output detection result for joint AU
    '''
    
    # first convolutional layer(+ activation layer)
    W_conv1 = weight_variable("W_conv1", [5, 5, 3, 32])
    b_conv1 = bias_variable("b_conv1", [32])
    h_conv1 = conv2d(x_image, W_conv1, b_conv1)#(, 64, 64, 32)
    h_pool1 = max_pool_2x2(h_conv1) #(None, 32, 32, 32)
    
    #second convolutional layer(+ activation layer)
    W_conv2 = weight_variable("W_conv2", [5, 5, 32, 64])
    b_conv2 = bias_variable("b_conv2", [64])
    h_conv2 = conv2d(h_pool1, W_conv2, b_conv2)#(None, 32, 32, 64)
    h_pool2 = max_pool_2x2(h_conv2) #(None, 16, 16, 64)
    
    #third convolutional layer(+ activation layer)
    W_conv3 = weight_variable("W_conv3", [3, 3, 64, 256])
    b_conv3 = bias_variable("b_conv3", [256])
    h_conv3 = conv2d(h_pool2, W_conv3, b_conv3) #(None, 16, 16, 64)
    h_pool3 = max_pool_2x2(h_conv3) #(None, 8, 8, 64)
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1", [8*8*256, 8192]) 
    b_fc1 = bias_variable("b_fc1", [8192])
    #reshape conv2 output to fit fully connected layer input
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*256]) 
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  
    h_fc1 = tf.nn.dropout(h_fc1,keep_prob)
    
    #output layer
    W_fc2 = weight_variable("W_fc2", [8192, 256]) 
    b_fc2 = bias_variable("b_fc2", [256])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    p_AUs = tf.nn.softmax(h_fc2)
    
    return p_AUs, h_fc2

def CNN_AUdetection_joint_5AU(x_image,keep_prob):
    '''
    three layer CNN for AU detection
    input should be one patch of images containing certain AUs
    output detection result for joint AU
    '''
    
    # first convolutional layer(+ activation layer)
    W_conv1 = weight_variable("W_conv1", [5, 5, 3, 32])
    b_conv1 = bias_variable("b_conv1", [32])
    h_conv1 = conv2d(x_image, W_conv1, b_conv1)#(, 64, 64, 32)
    h_pool1 = max_pool_2x2(h_conv1) #(None, 32, 32, 32)
    
    #second convolutional layer(+ activation layer)
    W_conv2 = weight_variable("W_conv2", [5, 5, 32, 64])
    b_conv2 = bias_variable("b_conv2", [64])
    h_conv2 = conv2d(h_pool1, W_conv2, b_conv2)#(None, 32, 32, 64)
    h_pool2 = max_pool_2x2(h_conv2) #(None, 16, 16, 64)
    
    #third convolutional layer(+ activation layer)
    W_conv3 = weight_variable("W_conv3", [3, 3, 64, 64])
    b_conv3 = bias_variable("b_conv3", [64])
    h_conv3 = conv2d(h_pool2, W_conv3, b_conv3) #(None, 16, 16, 64)
    h_pool3 = max_pool_2x2(h_conv3) #(None, 8, 8, 64)
    
    #fully connected layer
    W_fc1 = weight_variable("W_fc1", [8*8*64, 256]) 
    b_fc1 = bias_variable("b_fc1", [256])
    #reshape conv2 output to fit fully connected layer input
    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*64]) 
    
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  
    h_fc1 = tf.nn.dropout(h_fc1,keep_prob)
    
    #output layer
    W_fc2 = weight_variable("W_fc2", [256, 32]) 
    b_fc2 = bias_variable("b_fc2", [32])
    #reshape conv2 output to fit fully connected layer input
    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    p_AUs = tf.nn.softmax(h_fc2)
    
    return p_AUs, h_fc2


def Loss_AUModel_Labelonly(pred_au, label_au):
    #batch_size = tf.shape(pred_exp)[0]
    #loss_exp_gt = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label_exp, logits = pred_exp)) 
    
    #depth = tf.shape(pred_au)[1]
    one_au1 = tf.one_hot(label_au[:,0], 2)
    one_au2 = tf.one_hot(label_au[:,1], 2)
    one_au3 = tf.one_hot(label_au[:,2], 2)
    one_au4 = tf.one_hot(label_au[:,3], 2)
    one_au5 = tf.one_hot(label_au[:,4], 2)
    one_au6 = tf.one_hot(label_au[:,5], 2)
    one_au7 = tf.one_hot(label_au[:,6], 2)
    one_au8 = tf.one_hot(label_au[:,7], 2)
    one_au9 = tf.one_hot(label_au[:,8], 2)
    one_au10 = tf.one_hot(label_au[:,9], 2)
    one_au11 = tf.one_hot(label_au[:,10], 2)
    

    one_hot_label_au  = tf.concat([one_au1,one_au2,one_au3,one_au4,one_au5,one_au6,one_au7,one_au8,one_au9,one_au10,one_au11],axis=1) 
    loss_au_gt1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au1, logits = pred_au[:,0:2]))
    loss_au_gt2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au2, logits = pred_au[:,2:4]))
    loss_au_gt3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au3, logits = pred_au[:,4:6]))
    loss_au_gt4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au4, logits = pred_au[:,6:8]))
    loss_au_gt5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au5, logits = pred_au[:,8:10]))
    loss_au_gt6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au6, logits = pred_au[:,10:12]))
    loss_au_gt7 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au7, logits = pred_au[:,12:14]))
    loss_au_gt8 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au8, logits = pred_au[:,14:16]))
    loss_au_gt9 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au9, logits = pred_au[:,16:18]))
    loss_au_gt10 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au10, logits = pred_au[:,18:20]))
    loss_au_gt11 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au11, logits = pred_au[:,20:22]))
    
    loss_au_gt = loss_au_gt1 + loss_au_gt2 + loss_au_gt3 + loss_au_gt4 + loss_au_gt5 +loss_au_gt6 + loss_au_gt7 + loss_au_gt8 + loss_au_gt9 + loss_au_gt10 + loss_au_gt11
    return loss_au_gt

def Loss_AUModel_Labelonly_CK(pred_au, label_au):
    #batch_size = tf.shape(pred_exp)[0]
    #loss_exp_gt = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label_exp, logits = pred_exp)) 
    
    #depth = tf.shape(pred_au)[1]
    one_au1 = tf.one_hot(label_au[:,0], 2)
    one_au2 = tf.one_hot(label_au[:,1], 2)
    one_au3 = tf.one_hot(label_au[:,2], 2)
    one_au4 = tf.one_hot(label_au[:,3], 2)
    one_au5 = tf.one_hot(label_au[:,4], 2)
    one_au6 = tf.one_hot(label_au[:,5], 2)
    one_au7 = tf.one_hot(label_au[:,6], 2)
    one_au8 = tf.one_hot(label_au[:,7], 2)
    

    one_hot_label_au  = tf.concat([one_au1,one_au2,one_au3,one_au4,one_au5,one_au6,one_au7,one_au8],axis=1) 
    loss_au_gt1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au1, logits = pred_au[:,0:2]))
    loss_au_gt2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au2, logits = pred_au[:,2:4]))
    loss_au_gt3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au3, logits = pred_au[:,4:6]))
    loss_au_gt4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au4, logits = pred_au[:,6:8]))
    loss_au_gt5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au5, logits = pred_au[:,8:10]))
    loss_au_gt6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au6, logits = pred_au[:,10:12]))
    loss_au_gt7 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au7, logits = pred_au[:,12:14]))
    loss_au_gt8 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = one_au8, logits = pred_au[:,14:16]))
    
    loss_au_gt = loss_au_gt1 + loss_au_gt2 + loss_au_gt3 + loss_au_gt4 + loss_au_gt5 +loss_au_gt6 + loss_au_gt7 + loss_au_gt8
    return loss_au_gt


##################
def Loss_KnowledgeModel_joint(p_AUs, label_p_AUconfig, PGM_p_AUconfig, posterior_exp, coeff):
    
    # loss with GT expression
    # expected loss given ground truth expression
    loss_AU_gt = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = p_AUs, labels = label_p_AUconfig)) 
    
    # loss with all possible expressions
    num_configs = tf.shape(p_AUs)[1]
    num_exp = tf.shape(PGM_p_AUconfig)[0]
    Batch_size = tf.shape(p_AUs)[0]
    p_AUs_reshape = tf.reshape(tf.tile(tf.expand_dims(p_AUs, 1), [1, num_exp, 1]), [Batch_size*num_exp, num_configs])
    labels_reshape = tf.tile(PGM_p_AUconfig, [Batch_size, 1]) 
    loss_temp = tf.nn.softmax_cross_entropy_with_logits(logits = p_AUs_reshape, labels = labels_reshape)
    loss_temp_reshape = tf.reshape(loss_temp, [Batch_size, num_exp]) #(Batch_size, 5)
    loss_AU_posterior = tf.reduce_sum(tf.math.multiply(loss_temp_reshape, posterior_exp), 1) #(Batch_size, )
    loss_AU_posterior = tf.reduce_mean(loss_AU_posterior)
    
    loss_AUs = loss_AU_gt + coeff*loss_AU_posterior
    
    return loss_AUs

def Loss_KnowledgeModel_gtExpOnly_joint(p_AUs, label_p_AUconfig):
    
    # loss with GT expression
    # expected loss given ground truth expression
    loss_AU_gt = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = p_AUs, labels = label_p_AUconfig)) 
    
    return loss_AU_gt


########
def Loss_KnowledgeModel(p_AUs, list_AUconfig, label_p_AUconfig, PGM_p_AUconfig, posterior_exp, coeff):
    # loss with GT expression
    num_configs = tf.shape(list_AUconfig)[0]
    num_labels = tf.shape(p_AUs)[2]
    num_AUs = tf.shape(p_AUs)[1]
    num_exp = tf.shape(PGM_p_AUconfig)[0]
    Batch_size = tf.shape(p_AUs)[0]
    p_AUs_reshape = tf.reshape(tf.tile(tf.expand_dims(p_AUs, 1), [1, num_configs, 1, 1]), [Batch_size*num_configs, num_AUs, num_labels]) #(Batch_size*AU_configs, num_AUs, num_labels)
    p_AUs_reshape2 = tf.reshape(p_AUs_reshape, [Batch_size*num_configs*num_AUs, num_labels])
    labels_tile = tf.tile(list_AUconfig, [Batch_size, 1, 1]) #(Batch_size*AU_configs, num_AUs, num_labels)
    labels_reshape = tf.reshape(labels_tile, [Batch_size*num_configs*num_AUs, num_labels])
    
    loss_temp = tf.nn.softmax_cross_entropy_with_logits(logits = p_AUs_reshape2, labels = labels_reshape)
    loss_temp1 = tf.reshape(loss_temp, [Batch_size*num_configs, num_AUs])
    loss_temp2 = tf.reduce_sum(loss_temp1, 1) # sum loss for AU detectors
    loss_temp3 = tf.reshape(loss_temp2, [Batch_size, num_configs])
    loss_AU_gt = tf.reduce_sum(tf.math.multiply(loss_temp3, label_p_AUconfig), 1) # expected loss given ground truth expression
    loss_AU_gt = tf.math.reduce_mean(loss_AU_gt)
    
#     loss with all possible expressions
    loss_temp_reshape = tf.tile(tf.expand_dims(loss_temp3, 1), [1, num_exp, 1]) #(Batch_size, 5, 32)
    PGM_pAU_reshape = tf.tile(tf.expand_dims(PGM_p_AUconfig, 0), [Batch_size, 1,1]) #(Batch_size, 5, 32)
    loss_AU_exp = tf.reduce_sum(tf.math.multiply(loss_temp_reshape, PGM_pAU_reshape), 2) #(Batch_size, 5)
    
    loss_AU_posterior = tf.reduce_sum(tf.math.multiply(loss_AU_exp, posterior_exp), 1) #(Batch_size, )
    loss_AU_posterior = tf.reduce_mean(loss_AU_posterior)
    
    
    loss_AUs = loss_AU_gt + coeff*loss_AU_posterior
    
    return loss_AUs

def Loss_KnowledgeModel_gtExpOnly(p_AUs, list_AUconfig, label_p_AUconfig):
    # loss with GT expression
    num_configs = tf.shape(list_AUconfig)[0]
    num_labels = tf.shape(p_AUs)[2]
    num_AUs = tf.shape(p_AUs)[1]
    Batch_size = tf.shape(p_AUs)[0]
    p_AUs_reshape = tf.reshape(tf.tile(tf.expand_dims(p_AUs, 1), [1, num_configs, 1, 1]), [Batch_size*num_configs, num_AUs, num_labels]) #(Batch_size*AU_configs, num_AUs, num_labels)
    p_AUs_reshape2 = tf.reshape(p_AUs_reshape, [Batch_size*num_configs*num_AUs, num_labels])
    labels_tile = tf.tile(list_AUconfig, [Batch_size, 1, 1]) #(Batch_size*AU_configs, num_AUs, num_labels)
    labels_reshape = tf.reshape(labels_tile, [Batch_size*num_configs*num_AUs, num_labels])
    
    loss_temp = tf.nn.softmax_cross_entropy_with_logits(logits = p_AUs_reshape2, labels = labels_reshape)
    loss_temp1 = tf.reshape(loss_temp, [Batch_size*num_configs, num_AUs])
    loss_temp2 = tf.reduce_sum(loss_temp1, 1) # sum loss for AU detectors
    loss_temp3 = tf.reshape(loss_temp2, [Batch_size, num_configs])
    loss_AU_gt = tf.reduce_sum(tf.math.multiply(loss_temp3, label_p_AUconfig), 1) # expected loss given ground truth expression
    loss_AU_gt = tf.math.reduce_mean(loss_AU_gt)
    
    
    loss_AUs = loss_AU_gt
    
    return loss_AUs

def normalize(x, mean, var, beta, gamma):
    inv = tf.div(gamma, tf.sqrt(tf.add(var, 0.001)))
    return tf.add(tf.multiply(tf.subtract(x,mean),inv),beta)

def batch_normalization(x,phase_train,scope_name = "bn"):
    phase_train = tf.cast(phase_train,tf.bool)
    input_shape = x.shape

    if len(input_shape) == 4:
        n_out = input_shape[-1]
        moment_shape = [0,1,2]
    elif len(input_shape) ==2:
        n_out = 1
        moment_shape = [0]

    with tf.variable_scope(scope_name):
        beta = tf.get_variable("beta",initializer = tf.constant(0.0,shape = [n_out]), trainable=True)
        gamma = tf.get_variable("gamma", initializer = tf.constant(1.0,shape = [n_out]), trainable =True)
        batch_mean, batch_var = tf.nn.moments(x, moment_shape, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = normalize(x,mean,var,beta,gamma)
        return normed
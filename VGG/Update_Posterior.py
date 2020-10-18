# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:07:04 2019

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


def Exp_KnowledgeModel(P_Exp_AUs, AUs):
    '''
    given AU configuration provided by AU model
    compute the expression distribution given the AU configuration based on PGM
    
    '''
    ## convert binary to decimal to get index
    #power = tf.range(tf.cast(tf.shape(AUs)[1], dtype=tf.int64)) #[0,1,2,3,4]
    #multiply = tf.shape(AUs)[0]
    #power = tf.reshape(tf.tile(power, multiply))
    batch_size = tf.shape(AUs)[0]
    num_AUs = tf.shape(AUs)[1]
    expon = tf.tile(tf.transpose(tf.expand_dims(tf.range(tf.cast(num_AUs, dtype=tf.int32)),1)), [batch_size, 1])
    binary_base = tf.math.pow(2*tf.ones_like(AUs), expon)
    AU_decimal = tf.reduce_sum(tf.math.multiply(AUs, binary_base), axis=1)    
    p_exp_K = tf.gather(P_Exp_AUs, AU_decimal)
        
    return p_exp_K, AU_decimal


def Exp_KnowledgeModel_joint(P_Exp_AUs, p_AUs):
    '''
    given AU configuration distributions provided by AU model
    compute the expression distribution given the AU configuration distribution based on PGM
    
    '''
    Batch_size = tf.shape(p_AUs)[0]
    exp_num= tf.shape(P_Exp_AUs)[1]

    p_AUs_reshape = tf.tile(tf.expand_dims(p_AUs, 2), [1, 1, exp_num])#(None, 32, 5)
    PGM_exp_tile = tf.tile(tf.expand_dims(P_Exp_AUs, 0), [Batch_size, 1, 1]) #(None, 32, 5)    
    
    p_exp_K = tf.reshape(tf.reduce_sum(tf.math.multiply(p_AUs_reshape, PGM_exp_tile), axis=1), [Batch_size, exp_num]) #(None, 5) 
        
    return p_exp_K
    
def posterior_entropy(p_exp_ExpressionModel, p_exp_KnowledgeModel):
    '''
    given expression distribution provided by expression model and knowledge model
    combine these two distributions based on their entropy
    
    '''
    entropy_exp = tf.math.negative(tf.reduce_sum(tf.math.multiply(p_exp_ExpressionModel, tf.math.log(tf.clip_by_value(p_exp_ExpressionModel, 1e-10, 1))), 1))
    entropy_K = tf.math.negative(tf.reduce_sum(tf.math.multiply(p_exp_KnowledgeModel, tf.math.log(tf.clip_by_value(p_exp_KnowledgeModel, 1e-10, 1))), 1))
    
    #weight_exp = tf.math.reciprocal(entropy_exp)
    #weight_K = tf.math.reciprocal(entropy_K)
    
    weight_sum = tf.math.add(entropy_exp, entropy_K)
    weight_exp_norm = tf.math.divide(entropy_K, weight_sum)
    weight_K_norm = tf.math.divide(entropy_exp, weight_sum)
    num_exp_states = tf.shape(p_exp_ExpressionModel)[1]
    weight_e = tf.tile(tf.expand_dims(weight_exp_norm, 1), [1, num_exp_states])
    weight_k = tf.tile(tf.expand_dims(weight_K_norm, 1), [1, num_exp_states])
    posterior = tf.math.add(tf.math.multiply(weight_e, p_exp_ExpressionModel), tf.math.multiply(weight_k, p_exp_KnowledgeModel))
    
    return posterior, entropy_exp, entropy_K

def posterior_summation(p_exp_ExpressionModel, p_exp_KnowledgeModel):
    '''
    given expression distribution provided by expression model and knowledge model
    combine these two distributions based on their entropy
    
    '''
    
    p_exp_sum = tf.math.add(p_exp_ExpressionModel, p_exp_KnowledgeModel)
    p_exp_norm = tf.reduce_sum(p_exp_sum, axis=1)
    num_exp_states = tf.shape(p_exp_ExpressionModel)[1]
    p_exp_norm_tile = tf.tile(tf.expand_dims(p_exp_norm, axis=1), [1, num_exp_states])
    posterior = tf.math.divide(p_exp_sum, p_exp_norm_tile)
    
    return posterior
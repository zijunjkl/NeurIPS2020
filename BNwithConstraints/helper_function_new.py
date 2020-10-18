#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:03:05 2020

@author: zijun.cui
"""

import numpy as np
import tensorflow as tf
from sklearn.utils.extmath import cartesian
from scipy.optimize import fsolve
from random import seed
from random import random
from random import randint

def count_weights(states_arr, dag):
    num_nodes = len(states_arr)
    num_weights = np.zeros(num_nodes)
    for i in range(num_nodes):
        num_states = states_arr[i]
        if num_states == 2:
            num_weights[i] = sum(dag[:,i]) + 1
        else:
            num_weights[i] = num_states*(sum(dag[:,i]) + 1)
    
    return np.sum(num_weights), num_weights

def get_constraints(load_constraints): #multi-states nodes 
    Constraints = []
    num_constraints = load_constraints.shape[0]

    num_strict = 0
    for i in range(num_constraints):
        # target set
        orig_set = load_constraints[i,0]
        num_t = orig_set.shape[0]
        t_condi = []
        for j in range(num_t):
            t_condi.append([orig_set[j][0], orig_set[j][1]])
            
        # condition set
        orig_set = load_constraints[i,1]
        num_c = orig_set.shape[0]
        c_condi = []
        for j in range(num_c):
            if orig_set[j][0] == 18:
                c_condi.append([0, orig_set[j][1]-1]) # node 0 for expression with expression 0-5
            else:
                c_condi.append([orig_set[j][0], orig_set[j][1]])
            
        # 
        state = load_constraints[i,2][0]
        if state == 'strict bigger' or state == 'strict bigger-product':
            num_strict = num_strict + 1
            
        elif state == 'strict smaller' or state == 'strict smaller-product':
            num_strict = num_strict + 1
        #
        const = load_constraints[i,3]
        if const.shape[1] == 1:  # value as constraint
            prob = load_constraints[i,3]
            Constraints.append([t_condi,c_condi,state,prob[0,0]])
        else:
            num_cc = const.shape[0]
            cc_condi = []
            for j in range(num_cc):
                cc_condi.append([const[j][0], const[j][1]])
            Constraints.append([t_condi, c_condi, state, cc_condi])
            
    
    return Constraints, num_strict


def arrange_config_v2(config, DAG, states_arr):
    num_nodes = np.shape(DAG)[0]
    _, num_weights = count_weights(states_arr, DAG)
    num_config = np.shape(config)[0]
    
    num_weights = num_weights.astype(int)
    states_arr = states_arr.astype(int)
    
    parent_config = np.zeros([np.sum(states_arr), np.sum(num_weights), num_config])
    DAG = DAG + np.identity(num_nodes)
    
    for i in range(num_nodes):
        w_start_idx = np.sum(num_weights[0:i])
        w_end_idx = np.sum(num_weights[0:i+1])
        parent_set = np.nonzero(DAG[:,i])[0]
        if i == 0:
            pos = 0
        else:
            pos = np.sum(states_arr[0:i])
            
        if len(parent_set) != 0:
            if states_arr[i] == 2:
                for j in range(num_config):
#                    parent_config[pos+1, w_start_idx, j] = 1
                    temp = config[j, parent_set]
                    temp[i] = 0
                    parent_config[pos+1, w_start_idx:w_end_idx, j] = temp #config[j, parent_set]
            else: # multi-states
                for j in range(num_config):
                    temp = config[j, parent_set]
                    temp[i] = 0
                    ssize = (num_weights[i]/states_arr[i]).astype(int)
                    for k in range(states_arr[i]):
                        sidx = w_start_idx + k*ssize
                        eidx = w_start_idx + (k+1)*ssize
                        parent_config[pos+k, sidx:eidx, j] = temp
        else:
            if states_arr[i] == 2:
                for j in range(num_config):
                    parent_config[pos+1, w_start_idx:w_end_idx, j] = 0 #node itself
            else:
                for j in range(num_config):
                    for k in range(states_arr[i]):
                        sidx = w_start_idx + k
                        eidx = w_start_idx + k + 1
                        parent_config[pos+k, sidx:eidx, j] = 0
                        
        return parent_config
    
def complete_config(incomplete_config, states_arr):
    num_nodes = len(states_arr)
    
    idx = np.where(np.array(incomplete_config)==-999)[0]
    idx_complete = np.where(np.array(incomplete_config)!=-999)[0]
    
    num = len(idx)
    arr = []
    for i in range(num):
        arr.append(np.arange(states_arr[idx[i]]))
            
    config = cartesian(arr)
    num_config = np.shape(config)[0]
    
    complete_config = np.zeros([num_config, num_nodes])
    complete_config[:, idx] = config
    incomplete_config = np.array(incomplete_config)
    complete_config[:, idx_complete] = np.tile(incomplete_config[idx_complete], (num_config,1))
    
    return complete_config


def list_configuration(states_arr, dag):
    num_nodes = len(states_arr)
    
    configs = -999*np.ones(num_nodes)
    config = complete_config(configs, states_arr)
    config_p = arrange_config_v2(config, dag, states_arr)
    
    config = one_hot_config(config, states_arr)
    
    return config, config_p

def one_hot_config(config, states_arr):
    num_config, num_nodes = np.shape(config)
    states_cum = np.cumsum(states_arr)
    states_cum = np.insert(states_cum, 0, 0)
    num_states = np.sum(states_arr)
    num_states = num_states.astype(int)
    
    onehot_config = np.zeros([num_config, num_states])
    for i in range(num_config):
        for j in range(num_nodes):
            sidx = states_cum[j]
            ss = config[i,j]
            onehot_config[i, sidx.astype(int) + ss.astype(int)] = 1
    
    return onehot_config


def read_condition(constraint):
    num_condition = len(constraint)
    idx = []
    config = []
    for i in range(num_condition):
        idx.append(constraint[i][0])
        config.append(constraint[i][1])
        
    return idx, config


def sparse_constraint(Constraint, states_arr):
    num_constraints = len(Constraint)
    num_nodes = len(states_arr)
    
    configs = -999*np.ones(num_nodes)
    config_all = complete_config(configs, states_arr)
    num_config = config_all.shape[0]
    
    strict_idx = 0
    strict_const = []
    inequal_idx = 0
    inequal_const = []
    equal_idx = 0
    equal_const = []

    
    for i in np.arange(num_constraints):
        constraint = Constraint[i]
        # get target set and condition set
        target_idx, target_config = read_condition(constraint[0])
        condition_idx, condition_config = read_condition(constraint[1])
        
        # numerator
        mask_N = np.zeros([num_config, 1])
        mask_N[np.where((config_all[:,target_idx] == target_config) & \
               (config_all[:,condition_idx] == condition_config))[0]] = 1

        # denominator
        mask_D = np.zeros([num_config, 1])
        mask_D[np.where(config_all[:, condition_idx] == condition_config)] = 1
        
        if constraint[2] == 'bigger':
            if inequal_idx == 0:
                inequal_numer = -mask_N
                inequal_denom = mask_D
            else:
                inequal_numer = np.concatenate((inequal_numer, -mask_N), axis = 1)
                inequal_denom = np.concatenate((inequal_denom, mask_D), axis = 1)                
            inequal_const.append(-constraint[3])
            inequal_idx = inequal_idx + 1
            
        elif constraint[2] == 'smaller':
            if inequal_idx == 0:
                inequal_numer = mask_N
                inequal_denom = mask_D
            else:
                inequal_numer = np.concatenate((inequal_numer, mask_N), axis = 1)
                inequal_denom = np.concatenate((inequal_denom, mask_D), axis = 1)                
            inequal_const.append(constraint[3])
            inequal_idx = inequal_idx + 1
            
        elif constraint[2] == 'strict bigger':
            if strict_idx == 0:
                strict_numer = -mask_N
                strict_denom = mask_D
            else:
                strict_numer = np.concatenate((strict_numer, -mask_N), axis = 1)
                strict_denom = np.concatenate((strict_denom, mask_D), axis = 1)                
            strict_const.append(-constraint[3])
            strict_idx = strict_idx + 1
            
        elif constraint[2] == 'strict smaller':
            if strict_idx == 0:
                strict_numer = mask_N
                strict_denom = mask_D
            else:
                strict_numer = np.concatenate((strict_numer, mask_N), axis = 1)
                strict_denom = np.concatenate((strict_denom, mask_D), axis = 1)                
            strict_const.append(constraint[3])  
            strict_idx = strict_idx + 1
            
        elif constraint[2] == 'equal':
            if equal_idx == 0:
                equal_numer = mask_N
                equal_denom = mask_D
            else:
                equal_numer = np.concatenate((equal_numer, mask_N), axis = 1)
                equal_denom = np.concatenate((equal_denom, mask_D), axis = 1)                
            equal_const.append(constraint[3])   
            equal_idx = equal_idx + 1
            
    return strict_numer, strict_denom, strict_const, inequal_numer, inequal_denom, inequal_const,\
            equal_numer, equal_denom, equal_const



def sparse_constraint_v2(Constraint, states_arr):
    num_constraints = len(Constraint)
    num_nodes = len(states_arr)
    
    configs = -999*np.ones(num_nodes)
    config_all = complete_config(configs, states_arr)
    num_config = config_all.shape[0]
    
    strict2_idx = 0
    strict3_idx = 0

    
    for i in np.arange(num_constraints):
        constraint = Constraint[i]
        # get target set and condition set
        target_idx, target_config = read_condition(constraint[0])
        condition_idx, condition_config = read_condition(constraint[1])

        # numerator
        mask_N = np.zeros([num_config, 1])
        if len(condition_idx) != 0:
            temp1 = target_idx + condition_idx
            temp2 = target_config + condition_config
            idx = (config_all[:,temp1] == temp2).all(axis=1).nonzero()[0]
            mask_N[idx] = 1
        else:
            idx = (config_all[:,target_idx] == target_config).all(axis=1).nonzero()[0]
            mask_N[idx] = 1
            
        # denominator
        mask_D = np.zeros([num_config, 1])
        if len(condition_idx) != 0:
            idx = (config_all[:, condition_idx] == condition_config).all(axis=1).nonzero()[0]
            mask_D[idx] = 1
        
            
        if constraint[2] == 'strict bigger':
            
            const_idx, const_config = read_condition(constraint[3])
            # numerator for constraint
            mask_cN = np.zeros([num_config, 1])
            if len(condition_idx) != 0:
                temp1 = const_idx + condition_idx
                temp2 = const_config + condition_config
                idx = (config_all[:,temp1] == temp2).all(axis=1).nonzero()[0]
                mask_cN[idx] = 1
            else:
                idx = (config_all[:,const_idx] == const_config).all(axis=1).nonzero()[0]
                mask_cN[idx] = 1
    
    
            if strict2_idx == 0:
                strict_numer2 = -mask_N
                strict_denom2 = mask_D
                strict_const_numer2 = -mask_cN
            else:
                strict_numer2 = np.concatenate((strict_numer2, -mask_N), axis = 1)
                strict_denom2 = np.concatenate((strict_denom2, mask_D), axis = 1)     
                strict_const_numer2 = np.concatenate((strict_const_numer2, -mask_cN), axis = 1)
            strict2_idx = strict2_idx + 1
            
        elif constraint[2] == 'strict smaller':
            
            const_idx, const_config = read_condition(constraint[3])
            # numerator for constraint
            mask_cN = np.zeros([num_config, 1])
            if len(condition_idx) != 0:
                temp1 = const_idx + condition_idx
                temp2 = const_config + condition_config
                idx = (config_all[:,temp1] == temp2).all(axis=1).nonzero()[0]
                mask_cN[idx] = 1
            else:
                idx = (config_all[:,const_idx] == const_config).all(axis=1).nonzero()[0]
                mask_cN[idx] = 1
    
            if strict2_idx == 0:
                strict_numer2 = mask_N
                strict_denom2 = mask_D
                strict_const_numer2 = mask_cN
            else:
                strict_numer2 = np.concatenate((strict_numer2, mask_N), axis = 1)
                strict_denom2 = np.concatenate((strict_denom2, mask_D), axis = 1)     
                strict_const_numer2 = np.concatenate((strict_const_numer2, mask_cN), axis = 1)
            strict2_idx = strict2_idx + 1
        
        elif constraint[2] == 'strict bigger-product':
            
            const1_idx, const1_config = read_condition([constraint[3][0]])
            const2_idx, const2_config = read_condition([constraint[3][1]])
            # numerator for constraint
            mask_cN1 = np.zeros([num_config, 1])
            temp1 = const1_idx + condition_idx
            temp2 = const1_config + condition_config
            idx = (config_all[:,temp1] == temp2).all(axis=1).nonzero()[0]
            mask_cN1[idx] = 1
            
            mask_cN2 = np.zeros([num_config, 1])
            temp1 = const2_idx + condition_idx
            temp2 = const2_config + condition_config
            idx = (config_all[:,temp1] == temp2).all(axis=1).nonzero()[0]
            mask_cN2[idx] = 1
    
            if strict3_idx == 0:
                strict_numer3 = -mask_N
                strict_denom3 = mask_D
                strict_const_numer31 = -mask_cN1
                strict_const_numer32 = mask_cN2
            else:
                strict_numer3 = np.concatenate((strict_numer3, -mask_N), axis = 1)
                strict_denom3 = np.concatenate((strict_denom3, mask_D), axis = 1)     
                strict_const_numer31 = np.concatenate((strict_const_numer31, -mask_cN1), axis = 1)
                strict_const_numer32 = np.concatenate((strict_const_numer32, mask_cN2), axis = 1)
            strict3_idx = strict3_idx + 1
        
        elif constraint[2] == 'strict smaller-product':
            
            const1_idx, const1_config = read_condition([constraint[3][0]])
            const2_idx, const2_config = read_condition([constraint[3][1]])
            # numerator for constraint
            mask_cN1 = np.zeros([num_config, 1])
            temp1 = const1_idx + condition_idx
            temp2 = const1_config + condition_config
            idx = (config_all[:,temp1] == temp2).all(axis=1).nonzero()[0]
            mask_cN1[idx] = 1
            
            mask_cN2 = np.zeros([num_config, 1])
            temp1 = const2_idx + condition_idx
            temp2 = const2_config + condition_config
            idx = (config_all[:,temp1] == temp2).all(axis=1).nonzero()[0]
            mask_cN2[idx] = 1
    
            if strict3_idx == 0:
                strict_numer3 = mask_N
                strict_denom3 = mask_D
                strict_const_numer31 = mask_cN1
                strict_const_numer32 = mask_cN2
            else:
                strict_numer3 = np.concatenate((strict_numer3, mask_N), axis = 1)
                strict_denom3 = np.concatenate((strict_denom3, mask_D), axis = 1)     
                strict_const_numer31 = np.concatenate((strict_const_numer31, mask_cN1), axis = 1)
                strict_const_numer32 = np.concatenate((strict_const_numer32, mask_cN2), axis = 1)
            strict3_idx = strict3_idx + 1
            
    return strict_numer2, strict_denom2, strict_const_numer2, \
        strict_numer3, strict_denom3, strict_const_numer31, strict_const_numer32\
            
            

def update_slack(value, init):
    num_variable = len(init)
    update_value = value.copy()
    for i in np.arange(num_variable):
        func = lambda x: (value[i]+np.math.exp(x))*np.math.exp(x)/((value[i]+np.math.exp(x))**2+1) - x 
        update_value[i] = fsolve(func, init[i])
    
    return update_value



def check_satisfaction(strict_p, inequal_p, equal_p, strict_const_, inequal_const_, equal_const_):
    percent_strict = sum((strict_p-strict_const_)<0)/len(strict_p)
    percent_inequal = sum((inequal_p-inequal_const_)<0)/len(inequal_p)
    percent_equal = sum(abs(equal_p-equal_const_)<0.1)/len(equal_p)
    percent_total = (sum((strict_p-strict_const_)<0)+sum((inequal_p-inequal_const_)<0)+\
                     sum(abs(equal_p-equal_const_)<0.1))/(len(strict_p)+len(inequal_p)+len(equal_p))
    print('total = %f, strict = %f, inequal = %f, equal=%f'%(percent_total, percent_strict, percent_inequal, percent_equal))
    
'''tf'''

def WeightsToAdjacency(weights_r, states_cum):
    
    weights_pos = tf.nn.relu(weights_r)
    weights_neg = tf.nn.relu(-weights_r)
    weights = weights_pos + weights_neg
#    weights = tf.math.abs(weights_r)
    weights = tf.math.square(weights_r)
    
    row1 = tf.math.reduce_sum(weights_r[states_cum[0]:states_cum[1], :], axis=0, keepdims=True)    
    row2 = tf.math.reduce_sum(weights_r[states_cum[1]:states_cum[2], :], axis=0, keepdims=True)
    row3 = tf.math.reduce_sum(weights_r[states_cum[2]:states_cum[3], :], axis=0, keepdims=True)
    row4 = tf.math.reduce_sum(weights_r[states_cum[3]:states_cum[4], :], axis=0, keepdims=True)
    row5 = tf.math.reduce_sum(weights_r[states_cum[4]:states_cum[5], :], axis=0, keepdims=True)
    row6 = tf.math.reduce_sum(weights_r[states_cum[5]:states_cum[6], :], axis=0, keepdims=True)
    row7 = tf.math.reduce_sum(weights_r[states_cum[6]:states_cum[7], :], axis=0, keepdims=True)
    row8 = tf.math.reduce_sum(weights_r[states_cum[7]:states_cum[8], :], axis=0, keepdims=True)
    row9 = tf.math.reduce_sum(weights_r[states_cum[8]:states_cum[9], :], axis=0, keepdims=True)
    
    adjacency = tf.concat([row1, row2, row3, row4, row5, row6, row7, row8, row9], axis=0) #[num_states, num_config]
    
    return adjacency


def DAGconstraint(adjacency):
    '''
    (Zheng et al 2019)
    '''
    num_nodes = tf.shape(adjacency)[0]
    num_nodes = tf.dtypes.cast(num_nodes, tf.float32)
    E = tf.linalg.expm(adjacency*adjacency)
    h = tf.linalg.trace(E) - num_nodes 

    return h


def SPARSEconstraint_l1(adjacency):
    adj_ = adjacency
    
    return tf.math.reduce_sum(adj_)

def joint_marg_prob(states_cum, weights, bias, config, mask, parent_config):    
    num_states = tf.shape(parent_config)[0]
    num_config = tf.shape(config)[1]
    
    weight_reshape = tf.tile(tf.expand_dims(weights, 2), [num_states, 1, num_config])
    bias_reshape = tf.tile(bias, [1, num_config])
    
    val = tf.reduce_sum(tf.math.multiply(parent_config, weight_reshape), 1) + tf.math.multiply(bias_reshape, mask)
    
    dist1 = tf.nn.softmax(val[states_cum[0]:states_cum[1],:], axis=0)
    dist2 = tf.nn.softmax(val[states_cum[1]:states_cum[2],:], axis=0)
    dist3 = tf.nn.softmax(val[states_cum[2]:states_cum[3],:], axis=0)
    dist4 = tf.nn.softmax(val[states_cum[3]:states_cum[4],:], axis=0)
    dist5 = tf.nn.softmax(val[states_cum[4]:states_cum[5],:], axis=0)
    dist6 = tf.nn.softmax(val[states_cum[5]:states_cum[6],:], axis=0)
    dist7 = tf.nn.softmax(val[states_cum[6]:states_cum[7],:], axis=0)
    dist8 = tf.nn.softmax(val[states_cum[7]:states_cum[8],:], axis=0)
    dist9 = tf.nn.softmax(val[states_cum[8]:states_cum[9],:], axis=0)   

    distribution = tf.concat([dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9], axis=0) #[num_states, num_config]

    prob_arr = tf.math.pow(distribution, config)
    prob_arr2 = tf.math.reduce_prod(prob_arr, 0)  # joint probability for each configuration
    prob = tf.math.reduce_sum(distribution) # by lisiting all joint configurations, prob should always be 1 if DAG
    
    return prob, prob_arr2

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:35:37 2020

@author: zijun
"""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io
import os
import math
from scipy.optimize import fsolve
import random
import pickle
from helper_function_new import count_weights, list_configuration, get_constraints, joint_marg_prob, complete_config
from helper_function_new import WeightsToAdjacency, DAGconstraint, sparse_constraint, update_slack, check_satisfaction
import matplotlib.pyplot as plt


load_constraints = scipy.io.loadmat('Constraints_ExpDepSingleAU_AU6.mat')
load_constraints = load_constraints['Constraint']
num_constraints = load_constraints.shape[0]
Constraints, num_strict = get_constraints(load_constraints)

num_nodes = 9 
states_arr = 2*np.ones(num_nodes)
states_arr[0] = 6 #6 expressions
load_dag = np.ones(num_nodes) - np.identity(num_nodes) #ones for all off-diagonal elements

num_weights, _ = count_weights(states_arr, load_dag)
num_states = np.sum(states_arr)
num_states = np.sum(states_arr).astype(int)
num_states_cum = np.cumsum(states_arr)
num_states_cum = np.insert(num_states_cum, 0, 0).astype(int)

states_arr2 = np.zeros(num_nodes)
for i in np.arange(num_nodes):
    if states_arr[i] == 2:
        states_arr2[i] = 1
    else:
        states_arr2[i] = states_arr[i]
num_states2 = np.sum(states_arr2).astype(int)
num_states_cum2 = np.cumsum(states_arr2)
num_states_cum2 = np.insert(num_states_cum2, 0, 0).astype(int)

'''list all configurations'''
config_np, config_p_np = list_configuration(states_arr, load_dag)

strict_numer_, strict_denom_, strict_const_, inequal_numer_, inequal_denom_, inequal_const_,\
equal_numer_, equal_denom_, equal_const_ = sparse_constraint(Constraints, states_arr)

num_config = np.shape(config_np)[0]
mask_ = np.zeros([num_config, num_states])
for i in np.arange(num_nodes):
    pos = num_states_cum[i]
    if states_arr[i] == 2:
        mask_[:, pos+1] = 1
    else:
        mask_[:, pos:num_states_cum[i+1]] = 1
            
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #(e.g. it will use GPU’s number 0 and 2.)

tf.disable_v2_behavior()

'''input'''
config_list = tf.placeholder(tf.float32, shape=[num_states, None]) # one hot encoding of the configuration
mask_bias = tf.placeholder(tf.float32, shape=[num_states, None])
parent_config_list = tf.placeholder(tf.float32, shape=[num_states, num_weights, None])
states_cum = tf.placeholder(tf.int32, shape=[num_nodes+1, ])
states_cum2 = tf.placeholder(tf.int32, shape=[num_nodes+1, ])

strict_numer = tf.placeholder(tf.float32, shape=[num_config, None])
strict_denom = tf.placeholder(tf.float32, shape=[num_config, None])
strict_const = tf.placeholder(tf.float32, shape=[len(strict_const_),])
inequal_numer = tf.placeholder(tf.float32, shape=[num_config, None])
inequal_denom = tf.placeholder(tf.float32, shape=[num_config, None])
inequal_const = tf.placeholder(tf.float32, shape=[len(inequal_const_),])
equal_numer = tf.placeholder(tf.float32, shape=[num_config, None])
equal_denom = tf.placeholder(tf.float32, shape=[num_config, None])
equal_const = tf.placeholder(tf.float32, shape=[len(equal_const_),])

eta = tf.placeholder(tf.float32)
eta2 = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
alpha = tf.placeholder(tf.float32)
rho = tf.placeholder(tf.float32)
slack = tf.placeholder(tf.float32, shape=[num_strict, ])

'''variable'''
weights = tf.Variable(tf.truncated_normal([1,num_weights.astype(int)], stddev=0.5))
bias = tf.Variable(tf.truncated_normal([num_states,1], stddev=0.5))


'''joint probability'''
prob, prob_arr= joint_marg_prob(states_cum, weights, bias, config_list, mask_bias, parent_config_list)


'''constraints'''
prob_arr_s_n = tf.tile(tf.expand_dims(prob_arr, 1), [1, tf.shape(strict_numer)[1]])
prob_arr_s_d = tf.tile(tf.expand_dims(prob_arr, 1), [1, tf.shape(strict_denom)[1]])
prob_s_n = tf.reduce_sum(tf.math.multiply(strict_numer, prob_arr_s_n), axis=0)
prob_s_d = tf.reduce_sum(tf.math.multiply(strict_denom, prob_arr_s_d), axis=0)
strict_val = tf.math.divide(prob_s_n, prob_s_d)

prob_arr_i_n = tf.tile(tf.expand_dims(prob_arr, 1), [1, tf.shape(inequal_numer)[1]])
prob_arr_i_d = tf.tile(tf.expand_dims(prob_arr, 1), [1, tf.shape(inequal_denom)[1]])
prob_i_n = tf.reduce_sum(tf.math.multiply(inequal_numer, prob_arr_i_n), axis=0)
prob_i_d = tf.reduce_sum(tf.math.multiply(inequal_denom, prob_arr_i_d), axis=0)
inequal_val = tf.math.divide(prob_i_n, prob_i_d)

prob_arr_e_n = tf.tile(tf.expand_dims(prob_arr, 1), [1, tf.shape(equal_numer)[1]])
prob_arr_e_d = tf.tile(tf.expand_dims(prob_arr, 1), [1, tf.shape(equal_denom)[1]])
prob_e_n = tf.reduce_sum(tf.math.multiply(equal_numer, prob_arr_e_n), axis=0)
prob_e_d = tf.reduce_sum(tf.math.multiply(equal_denom, prob_arr_e_d), axis=0)
equal_val = tf.math.divide(prob_e_n, prob_e_d)


'''penalty'''
strict_penalty = tf.reduce_sum(tf.math.log((strict_val-strict_const+tf.math.exp(slack))**2 + 1))
inequal_penalty = tf.reduce_sum(tf.math.log((tf.nn.relu(inequal_val-inequal_const)**2 + 1)))
equal_penalty = tf.reduce_sum(tf.math.log((equal_val-equal_const)**2 + 1))
H = tf.dtypes.cast(tf.shape(strict_numer)[1], tf.float32)
J = tf.dtypes.cast(tf.shape(inequal_numer)[1], tf.float32)
K = tf.dtypes.cast(tf.shape(equal_numer)[1], tf.float32)
obj_penalty = strict_penalty/H + inequal_penalty/J + equal_penalty/K


'''DAG constraint'''
weights_re = tf.reshape(weights, [num_states2, num_nodes])
weights_re2 = WeightsToAdjacency(weights_re, states_cum2)
weights_prime = tf.matrix_set_diag(weights_re2, tf.zeros(weights_re2.shape[0], dtype = tf.float32))
adjacency = weights_prime #reads parents from each row
obj_dag = DAGconstraint(adjacency)

'''sparsity constraint'''
sparse = tf.norm(weights, ord=1)


'''total loss'''
obj = eta*obj_penalty + eta2*sparse + 0.5*rho*obj_dag*obj_dag + alpha*obj_dag


'''optimizer'''
opt = tf.train.AdamOptimizer(learning_rate).minimize(obj)

#sess = tf.Session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #config.gpu_options.per_process_gpu_memory_fraction = 0.85
sess = tf.Session(config=config)


'''hyper-parameters'''
eta_ = 20
eta2_ = 0.0001
step_size = 0.02
MaxIter = 100
MaxOutIter = 100
alpha_ = 0
rho_ = 1
best_loss = -10
flag = 0
nniter = 0
dag_tol = 1e-8

init = tf.global_variables_initializer()
sess.run(init)
slack_ = np.round(np.random.uniform(0,1,num_strict), 2)

for _ in range(5):
    for _ in range(MaxOutIter): #and flag == 0:
        while rho_ < 1e+20:
            feed = {states_cum2: num_states_cum2}
            loss_dag = sess.run(obj_dag, feed_dict=feed)        
            loss_dag_old = loss_dag
            
            old_loss = -1e5
            delta = 1e5
            niter = 0
            for niter in range(MaxIter): # number of epoch
                '''weights likelihood'''
                feed = {mask_bias:mask_.T, learning_rate:step_size, eta:eta_, eta2:eta2_, \
                        alpha:alpha_, rho:rho_, states_cum: num_states_cum,states_cum2: num_states_cum2, \
                        config_list:np.transpose(config_np), parent_config_list:config_p_np, \
                        strict_numer:strict_numer_, strict_denom:strict_denom_, strict_const:strict_const_, \
                        inequal_numer:inequal_numer_, inequal_denom:inequal_denom_, inequal_const:inequal_const_, \
                        equal_numer:equal_numer_, equal_denom:equal_denom_, equal_const:equal_const_, \
                        slack:slack_}
                _, total_loss, loss_penalty, loss_sparse, loss_dag, strict_p, inequal_p, equal_p, weights_,bias_, prob_arr_ = sess.run([opt, obj, obj_penalty, sparse, obj_dag, strict_val, inequal_val, equal_val, weights_re, bias, prob_arr], feed_dict=feed)        
                print('total_loss = %f, penalty_loss=%f, sparse=%f, dag_loss=%.8f'%(total_loss, loss_penalty, loss_sparse, loss_dag))
                delta = np.abs(total_loss - old_loss)
                old_loss = total_loss
                if delta < 2e-4:
                    break
                
            if loss_dag > 0.5*loss_dag_old:
                rho_ = rho_*5.0
            else:
                break
            
        alpha_ = alpha_ + rho_*loss_dag
        if loss_dag <= dag_tol:
            break
    
    check_satisfaction(strict_p, inequal_p, equal_p, strict_const_, inequal_const_, equal_const_)
    slack_ = update_slack(strict_p, slack_)
    

feed = {states_cum2: num_states_cum2}
adj = sess.run(adjacency, feed_dict=feed) 


fig, ax = plt.subplots(figsize=(7, 7))
colormap = plt.imshow(np.abs(adj), cmap = "cool")        
cbar = plt.colorbar(colormap, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=14)
ax.set_xticks([0,1,2,3,4,5,6,7,8])
ax.set_yticks([0,1,2,3,4,5,6,7,8])
x_label_list = ['Exp','AU1','AU2','AU4','AU6','AU7','AU12','AU15','AU17']
y_label_list = ['Exp','AU1','AU2','AU4','AU6','AU7','AU12','AU15','AU17']
#ax.set_xticklabels(x_label_list)
#ax.set_yticklabels(y_label_list)
axis_font = {'fontname':'Arial', 'size':'17'}
plt.xticks([0,1,2,3,4,5,6,7,8], x_label_list, **axis_font)
plt.yticks([0,1,2,3,4,5,6,7,8], y_label_list, **axis_font)
#plt.title('Structure with EI-JAU')
plt.gca().xaxis.tick_top()
plt.savefig('ED-SAU-new.pdf', bbox_inches='tight',dpi=300)

#fig, ax = plt.subplots(figsize=(6, 6))
#colormap = plt.imshow(np.abs(adj), cmap = "Blues_r")        
#cbar = plt.colorbar(colormap, fraction=0.046, pad=0.04)
#ax.set_xticks([0,1,2,3,4,5,6,7,8])
#ax.set_yticks([0,1,2,3,4,5,6,7,8])
#x_label_list = ['Exp','AU1','AU2','AU4','AU6','AU7','AU12','AU15','AU17']
#y_label_list = ['Exp','AU1','AU2','AU4','AU6','AU7','AU12','AU15','AU17']
#ax.set_xticklabels(x_label_list)
#ax.set_yticklabels(y_label_list)
#plt.title('Structure with ED-SAU')
#plt.gca().xaxis.tick_top()
#plt.savefig('ED-SAU-new.pdf')

#plt.figure()
#plt.imshow(weights_)
#configs = -999*np.ones(num_nodes)
#config = complete_config(configs, states_arr)
#scipy.io.savemat('./ExpDepSingle-1.mat',mdict={'adj':adj, 'weights':weights_, \
#                                             'bias':bias_, 'prob_arr':prob_arr_,'config_list':config})


# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:45:50 2019

@author: Zijun Cui
"""

import tensorflow as tf
from helper_functions import weight_variable, bias_variable, conv2d, max_pool_2x2, get_train_batch, get_valid_test_set
from helper_functions import Compute_F1score, Compute_F1score_au
from AU_Knowledge_part_new import CNN_AUdetection_joint, Loss_KnowledgeModel_joint, Loss_KnowledgeModel_gtExpOnly_joint
from Update_Posterior import Exp_KnowledgeModel_joint
from ThirdModel_part import Loss_3rdModel, Prediction_3rdModel_joint
import pickle
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 50
AU_config_num = 256 # 2^5
Exp_config_num = 5

tf.reset_default_graph()

'''  
input
'''
label_Expression = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name = 'label_Expression')

# for 3rd/PGM right expression model
p_AUs_fix = tf.compat.v1.placeholder(tf.float32, shape=[None, AU_config_num], name='p_AUs_fix') # p({Zi=on}|X) 3rd model


PGM_p_Exp = tf.compat.v1.placeholder(tf.float32, shape=[AU_config_num, Exp_config_num], name='PGM_p_Exp') # p(Y|{Zi})

learning_rate = tf.placeholder(tf.float32, name='learning_rate')
'''
models
'''    

#output of Knowledge model
p_Exp_K = Exp_KnowledgeModel_joint(PGM_p_Exp, p_AUs_fix) #(None, 5)
pred_Exp_K = tf.math.argmax(p_Exp_K, axis = 1)

#output of Third Model(Left expression model)
p_Exp_3rdModel, fc_3rd = Prediction_3rdModel_joint(p_AUs_fix) #(None, 32)
pred_Exp_3rd = tf.math.argmax(p_Exp_3rdModel, axis = 1)

#loss for third model(left expression model)
loss_3rd = Loss_3rdModel(label_Expression, p_Exp_3rdModel)

'''
train
'''
gpu_options = tf.GPUOptions(allow_growth = True)

opt_3rd = tf.train.AdamOptimizer(learning_rate)
train_3rd = opt_3rd.minimize(loss_3rd)

tf.add_to_collection('activation', p_Exp_K)
tf.add_to_collection('activation', pred_Exp_K)
tf.add_to_collection('activation', p_Exp_3rdModel)
tf.add_to_collection('activation', pred_Exp_3rd)
tf.add_to_collection('activation', loss_3rd)
tf.add_to_collection('activation', train_3rd)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

file_name = 'define_right_exp'
modelFolder = os.path.join( os.getcwd(), 'Models') 
model_path = os.path.join(modelFolder, file_name)
#model_path = os.path.join('E:/Models', file_name)
save_path = saver.save(sess, model_path)
print('Model is saved in path: %s' % save_path)

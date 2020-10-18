# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:48:54 2019

@author: Zijun Cui
"""
import tensorflow as tf
from helper_functions import weight_variable, bias_variable, conv2d, max_pool_2x2, get_train_batch, get_valid_test_set
from helper_functions import Compute_F1score, Compute_F1score_au
from AU_Knowledge_part_new import CNN_AUdetection_joint_8AU, Loss_KnowledgeModel_joint, Loss_KnowledgeModel_gtExpOnly_joint
from Update_Posterior import Exp_KnowledgeModel_joint
from ThirdModel_part import Loss_3rdModel, Prediction_3rdModel_joint

import pickle
import os

## for CK+ 
im_width = 64
im_height = 64
AU_number = 8
AU_config_num = 256 # 2^5
Exp_config_num = 6

tf.reset_default_graph()

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''  
input
'''
x_image_orig = tf.compat.v1.placeholder(tf.float32, shape=[None, im_height, im_width, 3], name='x_image_orig')
x_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x_image_orig)
#    label_Expression = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name = 'label_expression')

# for AU model training
label_p_AUconfig = tf.compat.v1.placeholder(tf.float32, shape=[None, AU_config_num], name='label_p_AUconfig') # p({Zi}|Exp=GT)
PGM_p_AUconfig  = tf.compat.v1.placeholder(tf.float32, shape=[Exp_config_num, AU_config_num], name='PGM_p_AUconfig')
list_AUconfig = tf.compat.v1.placeholder(tf.int32, shape=[AU_config_num, AU_number], name='list_AUconfig') # list all possiblt AU configurations {Zi} with one-hot encoding

posterior = tf.compat.v1.placeholder(tf.float32, shape=[None, Exp_config_num], name='posterior')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
balance_w = tf.placeholder(tf.float32, name='balance_w')

'''
models
'''    
#output of AU detection model
p_AUs, fc_AUs = CNN_AUdetection_joint_8AU(x_image, keep_prob)

#predict AU states
pred_AU_joint = tf.math.argmax(p_AUs, axis = 1) #(None, 1)
pred_AUs = tf.gather(list_AUconfig,  pred_AU_joint)

#loss for AU model
loss_AU_phase1 = Loss_KnowledgeModel_gtExpOnly_joint(fc_AUs, label_p_AUconfig)
loss_AU_phase2 = Loss_KnowledgeModel_joint(fc_AUs, label_p_AUconfig, PGM_p_AUconfig, posterior, balance_w)

'''
train
'''
gpu_options = tf.GPUOptions(allow_growth = True)

opt_AU = tf.train.AdamOptimizer(learning_rate)
train_AU_phase1 = opt_AU.minimize(loss_AU_phase1)
train_AU_phase2 = opt_AU.minimize(loss_AU_phase2)

tf.add_to_collection('activation', p_AUs)
tf.add_to_collection('activation', pred_AUs)
tf.add_to_collection('activation', loss_AU_phase1)
tf.add_to_collection('activation', train_AU_phase1)
tf.add_to_collection('activation', loss_AU_phase2)
tf.add_to_collection('activation', train_AU_phase2)

# save the model
saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

file_name = 'define_AU_p2_8AU'
model_folder = os.path.join(os.path.join(os.getcwd()), 'Models')
model_path = os.path.join(model_folder, file_name)
#model_path = os.path.join('E:/Models', file_name)
save_path = saver.save(sess, model_path)
print('Model is saved in path: %s' % save_path)
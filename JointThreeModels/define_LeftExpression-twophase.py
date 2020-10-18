# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:56:27 2019

@author: Zijun Cui
"""
import tensorflow as tf
from helper_functions import weight_variable, bias_variable, conv2d, max_pool_2x2, get_train_batch, get_valid_test_set
from helper_functions import Compute_F1score, Compute_F1score_au
from Update_Posterior import Exp_KnowledgeModel_joint
from Expression_part_new import CNN_Expression, Loss_ExpressionModel, Loss_ExpressionModel_Labelonly
import pickle
import os

im_width = 64
im_height = 64
AU_number = 5
AU_config_num = 32 # 2^5
Exp_config_num = 5

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  #(e.g. it will use GPU’s number 0 and 2.)

'''  
input
'''
x_image_orig = tf.compat.v1.placeholder(tf.float32, shape=[None, im_height, im_width, 3], name='x_image_orig')
x_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x_image_orig)

label_Expression = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name = 'label_Expression')
posterior = tf.compat.v1.placeholder(tf.float32, shape=[None, Exp_config_num], name='posterior')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
balance_e = tf.placeholder(tf.float32, name='balance_e')

'''
models
'''

#output of Expression model
p_Expression, feat_output, regular = CNN_Expression(x_image, keep_prob) #(None, 5)
pred_Expression = tf.math.argmax(p_Expression, axis = 1)


#loss for Expression model
temp1 = Loss_ExpressionModel_Labelonly(feat_output, label_Expression)
loss_exp_phase1 = temp1 + 0.0001*regular

temp2 = Loss_ExpressionModel(feat_output, label_Expression, posterior, balance_e)
loss_exp_phase2 = temp2 + 0.0001*regular

'''
train
'''
opt_Exp = tf.train.AdamOptimizer(learning_rate)
train_Exp_phase1 = opt_Exp.minimize(loss_exp_phase1)
train_Exp_phase2 = opt_Exp.minimize(loss_exp_phase2)

tf.add_to_collection('activation', p_Expression)
tf.add_to_collection('activation', pred_Expression)
tf.add_to_collection('activation', loss_exp_phase1)
tf.add_to_collection('activation', train_Exp_phase1)
tf.add_to_collection('activation', loss_exp_phase2)
tf.add_to_collection('activation', train_Exp_phase2)

# save the model
saver = tf.train.Saver()

gpu_options = tf.GPUOptions(allow_growth = True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

file_name = 'define_LeftExp_p2'
model_folder = os.path.join(os.path.join(os.getcwd()), 'Models')
model_path = os.path.join(model_folder, file_name)
#model_path = os.path.join('E:/Models', file_name)
save_path = saver.save(sess, model_path)
print('Model is saved in path: %s' % save_path)
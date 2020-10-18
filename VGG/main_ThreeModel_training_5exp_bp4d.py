# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:54:08 2019
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
from helper_functions import weight_variable, bias_variable, conv2d, max_pool_2x2, get_train_batch_ck, get_valid_test_set,get_train_batch
from helper_functions import posterior_entropy, Compute_F1score, Compute_F1score_au, posterior_entropy_three_2, posterior_entropy_three
from AU_Knowledge_part_new import CNN_AUdetection_joint, Loss_KnowledgeModel_joint, Loss_KnowledgeModel_gtExpOnly_joint
from Update_Posterior import posterior_summation, Exp_KnowledgeModel_joint
from ThirdModel_part import Loss_3rdModel, Prediction_3rdModel_joint
from ImportGraph_twophase import ImportRightAU, ImportRightExp, ImportLeftExp,ImportLeftExp_vgg,ImportRightExp_BP4D
#from Ipython.display import Image, display
import vgg19_trainable as vgg19
import utils
import itertools
import scipy.io as sio
from Expression_part_new import CNN_Expression, Loss_ExpressionModel, Loss_ExpressionModel_Labelonly
import pdb
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #(e.g. it will use GPU’s number 0 and 2.)


#training, testing index
pickleFolder = os.path.join( os.getcwd(), 'data_pickle') 
#pickle_in = open( os.path.join( pickleFolder, '3fold-image-serverpath.p'), "rb" )

data = sio.loadmat(os.path.join( pickleFolder,'BP4D_Apex_11AU.mat'))
index = data['BP4D_Apex_11AU']
sub_name = index[0,0]['SUB']
task_name = index[0,0]['TASK']
image_name = index[0,0]['IMGIND']
label_exp = index[0,0]['EXP']
label_AU = index[0,0]['AU']
path_all=[]
for i in range(sub_name.shape[0]):
    if sub_name[i] // 1000 == 1:
#        path = ["E:/BP4D_data/color_64/F%03d_"%(sub_name[i]%1000)+"T%d_"%(task_name[i])+"%04d.jpg"%(image_name[i])]
        path = ["/training_data/zijun/color_64/F%03d_"%(sub_name[i]%1000)+"T%d_"%(task_name[i])+"%04d.jpg"%(image_name[i])]
    if sub_name[i] // 1000 == 2:
#        path = ["E:/BP4D_data/color_64/M%03d_"%(sub_name[i]%1000)+"T%d_"%(task_name[i])+"%04d.jpg"%(image_name[i])]
        path = ["/training_data/zijun/color_64/M%03d_"%(sub_name[i]%1000)+"T%d_"%(task_name[i])+"%04d.jpg"%(image_name[i])]
    if i == 0:
        path_all=path
    else:
        path_all=path_all + path

label_ = sio.loadmat('BP4D_8AU.mat')
label_AU = label_['AU_8']
fold1_index=path_all[0:248]
fold2_index=path_all[248:492]
fold3_index=path_all[492:732]
fold1_label=np.concatenate((label_exp[0:248], label_AU[0:248, :] ), axis = 1) #5AU index [0,1,5,6,9]
fold2_label=np.concatenate((label_exp[248:492], label_AU[248:492, :] ), axis = 1)
fold3_label=np.concatenate((label_exp[492:732], label_AU[492:732, :] ), axis = 1)


def main(fold1,fold2,fold3, foldL1,foldL2,foldL3 , Modelname, rightau, rightexp, leftexp):
    trainD_path = np.concatenate([fold1,fold2],axis=0)
    testD_path = fold3
    trainL = np.concatenate([foldL1,foldL2],axis=0)
    testL = foldL3

    pickle_in = open(os.path.join( '/home/zijun/Documents/BP4D/data_pickle/BP4D-8AU-5Exp/'+'list_AU_config.p'), "rb" )
    load_AUconfig = pickle.load(pickle_in)
    
    pickle_in = open( os.path.join( '/home/zijun/Documents/BP4D/data_pickle/BP4D-8AU-5Exp/'+ 'PGM_p-K2-Bayes.p'), "rb" )
    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)

#    pickle_in = open(os.path.join( '/home/zijun/Documents/PseudoBN/5exp-8AU/mysoln/'+'list_AU_config.p'), "rb" )
#    load_AUconfig = pickle.load(pickle_in)
#    
#    pickle_in = open( os.path.join( '/home/zijun/Documents/PseudoBN/5exp-8AU/mysoln/'+ 'PGM_p-K2-Bayes.p'), "rb" )
#    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    
    print('finish loading BN')
    epo = 20
    out_epo = 40
    batch_size = 64
    
    perf_exp_training = []
    perf_exp_testing = []
    
    perf_au_training = []
    perf_au_testing = []
    
    perf_posterior = []
    
    total_samples = len(trainD_path)
    
#    model_folder = os.path.join(os.path.join(os.getcwd()), 'Models')
    model_folder = '/home/zijun/Documents/BP4D/Models'
    rightAU_model_path = os.path.join(model_folder, rightau)
    right_AU_model = ImportRightAU(rightAU_model_path)
    
    model_folder2 = '/home/zijun/Documents/BP4D/Models'
    rightExp_model_path = os.path.join(model_folder2, rightexp)
    right_Exp_model = ImportRightExp_BP4D(rightExp_model_path)
    #left expression:
    graph = tf.Graph()	        
    sess = tf.Session(graph=graph)	   
    print(Modelname)
    print('start')     
    with graph.as_default():	            
         # Import saved model from location 'loc' into local graph	            
         #vgg 
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        images_norm = tf.map_fn(lambda frame:tf.image.per_image_standardization(frame),images)
        true_out = tf.placeholder(tf.int32, [None, ])
        train_mode = tf.placeholder(tf.bool)
        balance_e_ = tf.placeholder(tf.float32, name='balance_e')
        posterior_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 5], name='posterior')
#        vgg = vgg19.Vgg19('/home/zijun/Documents/VGG-model/ThreeModels_withCNN/save_vgg/LeftExp-fold1_pre_ck.npy')
        vgg = vgg19.Vgg19(os.path.join( '/home/zijun/Documents/VGG-model/ThreeModels_withCNN/save_vgg/'+ Modelname))
        vgg.build(images_norm, train_mode)
#        print(vgg.get_var_count())
        sess.run(tf.global_variables_initializer())
        loss_exp = Loss_ExpressionModel_Labelonly(vgg.prob_exp, true_out)
        
        pred_Expression = tf.math.argmax(vgg.prob_exp, axis = 1)
        loss2 = Loss_ExpressionModel(vgg.prob_exp, true_out, posterior_, balance_e_)
#        loss_exp = loss2
        
        learn_rate=0.01
        train_vgg = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_exp)
        train_vgg2 = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss2)
    #leftExp_model_path = os.path.join(model_folder, leftexp)
    #left_Exp_model = ImportLeftExp(leftExp_model_path)
    
    balance_e = 0.001 #for left expression
    balance_w = 0.0005 #for right AU
    
    best_left = -10
    best_post2 = -10
    best_post3 = -10
    
    print('start')
    for out_e in range(out_epo):
            
        train_image, train_explabel, train_AUlabel, train_AUprob, train_image_224 = get_train_batch(900, trainD_path, trainL, load_PGM_pAUconfig, 0)
        #p_exp_left, _, _ = left_Exp_model.run(train_image, train_explabel)
        f_size=train_image.shape[0]
#        pdb.set_trace()
        temp_pred1 = sess.run(vgg.prob_exp, feed_dict={images: train_image_224[0:100,:,:], true_out: train_explabel[0:100], train_mode: False})
#        pdb.set_trace()
        temp_pred2 = sess.run(vgg.prob_exp, feed_dict={images: train_image_224[100:200,:,:], true_out: train_explabel[100:200], train_mode: False})
        temp_pred3 = sess.run(vgg.prob_exp, feed_dict={images: train_image_224[200:f_size,:,:], true_out: train_explabel[200:f_size], train_mode: False})
        
        temp_pred=np.concatenate([temp_pred1,temp_pred2, temp_pred3],axis=0)
        p_au, _, _ = right_AU_model.run(train_image, train_AUprob, load_AUconfig)
        p_exp_K, _, p_exp_right, _, _ = right_Exp_model.run(p_au, train_explabel, load_PGM_pExp)
        posterior_train = posterior_entropy(p_exp_right, temp_pred)
        
        test_image, test_explabel, test_AUlabel, test_AUprob, test_image_224 = get_train_batch(900, testD_path, testL, load_PGM_pAUconfig, 0)
        #p_exp_left, _, _ = left_Exp_model.run(test_image, test_explabel)
        temp_pred = sess.run(vgg.prob_exp, feed_dict={images: test_image_224, true_out: test_explabel, train_mode: False})
        
        p_au, _, _ = right_AU_model.run(test_image, test_AUprob, load_AUconfig)
        p_exp_K, _, p_exp_right, test_pred_3rd, _ = right_Exp_model.run(p_au, test_explabel, load_PGM_pExp)
        
        acc_right_exp =  (np.ravel(testL[:,0]-1)== test_pred_3rd).sum()/len(testL)
        print('acc right exp testing')
        print(acc_right_exp)
        
        posterior_test = posterior_entropy(p_exp_right, temp_pred)
        posterior_test2 = posterior_entropy_three(p_exp_right, temp_pred, p_exp_K)
        acc_train = (np.ravel(trainL[:,0]-1)== np.argmax(posterior_train, axis=1)).sum()/len(trainL)
        acc_test = (np.ravel(testL[:,0]-1)== np.argmax(posterior_test, axis=1)).sum()/len(testL)
        acc_test2 = (np.ravel(testL[:,0]-1)==np.argmax(posterior_test2, axis=1)).sum()/len(testL)
        perf_posterior.append([acc_train, acc_test])
        print('----------Outer_epo %d Posterior Acc Train %f Test(2) %f Test(3) %f-------------' %(out_e, acc_train, acc_test, acc_test2))
        print(' ')
        if acc_test > best_post2:
            best_post2 = acc_test
        if acc_test2 > best_post3:
            best_post3 = acc_test2
        
        LR = 0.0001
        for e in range(epo): #each epo go over all samples
            
            '''performance on testing for each epo'''
            start_idx_test = np.arange(0, len(testD_path), 900)
    #        for ti in range(len(start_idx_test)): 
            test_image, test_explabel, test_AUlabel, test_AUprob, test_image_224= get_train_batch(900, testD_path, testL, load_PGM_pAUconfig, start_idx_test[0])
            
            posterior = posterior_test
            
            p_au, _, _ = right_AU_model.run(test_image, test_AUprob, load_AUconfig)
            # left exp output
            #_, pred_exp_left, loss_left = left_Exp_model.run_p2(test_image, test_explabel, posterior, balance_e)
            loss_left, pred_exp_left = sess.run([loss2, pred_Expression], feed_dict={images: test_image_224, true_out: test_explabel, train_mode: False, posterior_:posterior, balance_e_: balance_e  })
            #loss_left=0
            # right AU output
            _, pred_au, loss_au = right_AU_model.run_p2(test_image, test_AUprob, load_AUconfig, load_PGM_pAUconfig, posterior, balance_w)
            # right exp /PGM model
            _, pred_exp_K, _, pred_exp_right, loss_right = right_Exp_model.run(p_au, test_explabel, load_PGM_pExp)
            
            acc_left = (np.ravel(testL[:,0]-1)== pred_exp_left).sum()/len(testL)
            f1_left =0
            acc_right = (np.ravel(testL[:,0]-1)== pred_exp_right).sum()/len(testL)
            f1_right =0
            acc_K = (np.ravel(testL[:,0]-1)== pred_exp_K).sum()/len(testL)
            f1_K =0
            
            f1 = Compute_F1score_au(test_AUlabel, pred_au)
#            pdb.set_trace()
#            f1 = 0
    #        perf_exp_testing.append([acc_left, f1_left, acc_right, f1_right, acc_K, f1_K, acc_posterior, f1_posterior, loss_left, loss_right, loss_au])
#            perf_exp_testing.append([acc_left, f1_left, acc_right, f1_right, acc_K, f1_K, loss_left, loss_right, loss_au])
#            perf_au_testing.append(f1)
#            print('Loss left exp || right exp || au')
#            print(loss_left, loss_right, loss_au)
#            print(' ')
            print('testing')
            print('exp acc: left || right || PGM')
            print(acc_left, acc_right, acc_K)
            if acc_left > best_left:
                best_left = acc_left
            
            start_idx = np.arange(0, total_samples, batch_size)         
            for i in range(len(start_idx)): 
                
                index = start_idx[i]
                train_image, train_explabel, train_AUlabel, train_AUprob,train_image_224 = get_train_batch(batch_size, trainD_path, trainL, load_PGM_pAUconfig, index)

                posterior = posterior_train[index:(batch_size+index), :]
                
                p_au, _, _ = right_AU_model.run(train_image, train_AUprob, load_AUconfig)
                #train left exp
                #left_Exp_model.train_p2(train_image, train_explabel, posterior, balance_e, LR)

                loss_left,_, pred_exp_left = sess.run([loss2,train_vgg2, pred_Expression], feed_dict={images: train_image_224, true_out: train_explabel, train_mode: True, posterior_:posterior, balance_e_: balance_e  })
                #train right exp 
                right_Exp_model.train(p_au, train_explabel, LR, load_PGM_pExp)

                #train right AU
                right_AU_model.train_p2(train_image, train_AUprob, LR, load_AUconfig, load_PGM_pAUconfig, posterior, balance_w)

            
#            '''performance on testing for each epo'''
#            start_idx_test = np.arange(0, len(testD_path), 900)
#    #        for ti in range(len(start_idx_test)): 
#            test_image, test_explabel, test_AUlabel, test_AUprob,test_image_224  = get_train_batch(900, testD_path, testL, load_PGM_pAUconfig, start_idx_test[0])
#            
#            posterior = posterior_test
#            
#            p_au, _, _ = right_AU_model.run(test_image, test_AUprob, load_AUconfig)
#            # left exp output
#            #_, pred_exp_left, loss_left = left_Exp_model.run_p2(test_image, test_explabel, posterior, balance_e)
#            loss_left, pred_exp_left = sess.run([loss2, pred_Expression], feed_dict={images: test_image_224, true_out: test_explabel, train_mode: False, posterior_:posterior, balance_e_: balance_e  })
#            
#            # right AU output
#            _, pred_au, loss_au = right_AU_model.run_p2(test_image, test_AUprob, load_AUconfig, load_PGM_pAUconfig, posterior, balance_w)
#            # right exp /PGM model
#            _, pred_exp_K, _, pred_exp_right, loss_right = right_Exp_model.run(p_au, test_explabel, load_PGM_pExp)
#        
#            acc_left = (np.ravel(testL[:,0]-1)== pred_exp_left).sum()/len(testL)
#            f1_left = 0
#            acc_right = (np.ravel(testL[:,0]-1)== pred_exp_right).sum()/len(testL)
#            f1_right = 0
#            acc_K = (np.ravel(testL[:,0]-1)== pred_exp_K).sum()/len(testL)
#            f1_K = 0
#    #        acc_posterior = (np.ravel(testL[:,0]-1)== pred_posterior).sum()/len(testL)
#    #        f1_posterior = Compute_F1score(testL[:,0]-1, pred_posterior)
#            
#            f1 = Compute_F1score_au(test_AUlabel, pred_au)
##            f1 = 0
#    #        perf_exp_testing.append([acc_left, f1_left, acc_right, f1_right, acc_K, f1_K, acc_posterior, f1_posterior, loss_left, loss_right, loss_au])
#            perf_exp_testing.append([acc_left, f1_left, acc_right, f1_right, acc_K, f1_K, loss_left, loss_right, loss_au])
#            perf_au_testing.append(f1)
#            
#            print('testing')
#            print('exp acc: left || right || PGM')
#            print(acc_left, acc_right, acc_K)

       
    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, Modelname), "wb" )
    pickle.dump([best_left, best_post2, best_post3], pickle_out)
    
    pickle_out.close()
    
 
#main(fold1_index, fold2_index, fold3_index, fold1_label, fold2_label, fold3_label,'LeftExp-fold1_pre_bp4d_3rd.npy','initialAU-joint-fold1-new',  'initialRightExp-joint-fold1-new', 'left_exp_5fold_1')
#main(fold1_index, fold3_index, fold2_index, fold1_label, fold3_label, fold2_label,'LeftExp-fold2_pre_bp4d_3rd.npy','initialAU-joint-fold2-new',  'initialRightExp-joint-fold2-new', 'left_exp_5fold_2')
main(fold2_index, fold3_index, fold1_index, fold2_label, fold3_label, fold1_label,'LeftExp-fold3_pre_bp4d_3rd.npy','initialAU-joint-fold3-new',  'initialRightExp-joint-fold3-new', 'left_exp_5fold_3')

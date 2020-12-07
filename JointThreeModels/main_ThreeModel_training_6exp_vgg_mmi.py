# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:54:08 2019
@author: Zijun Cui
"""

import numpy as np
import tensorflow as tf
import pickle
import os
from helper_functions import get_train_batch
from helper_functions import posterior_entropy, Compute_F1score_au
from ImportGraph_twophase import ImportRightAU, ImportRightExp

import vgg19_trainable as vgg19
import scipy.io as sio
from Expression_part_new import Loss_ExpressionModel, Loss_ExpressionModel_Labelonly
import pdb

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #(e.g. it will use GPU’s number 0 and 2.)

data = sio.loadmat(os.path.join('./MMI_8AU_6Exp.mat'))
index = data['MMI_8AU_6Exp']
sub_name = index[0,0]['SUB']
session_name = index[0,0]['Session']
frame_name = index[0,0]['Frame']
label_exp = index[0,0]['EXP']
label_AU = index[0,0]['AU']
path_all=[]
for i in range(sub_name.shape[0]):
    path = ["../Dataset/MMI/%d-"%(session_name[i])+"%d"%(frame_name[i])+".jpg"]
    if i == 0:
        path_all=path
    else:
        path_all=path_all + path

f1 = [28,30,31,32,33] #
f2 = [34,35,48,37,40]  #96
f3 = [38,39,47,42,43,44]  #111
f4 = [45,36,49]
f5 = [46,41,50]

fold1_index = []
fold2_index = []
fold3_index = []
fold4_index = []
fold5_index = []
fold1_label = []
fold2_label = []
fold3_label = []
fold4_label = []
fold5_label = []
for i in np.arange(504):
    S = sub_name[i]
    temp = np.zeros(9)
    temp[0] = label_exp[i,0]
    temp[1:] = label_AU[i,:]
    if S in f1:
        fold1_index.append(path_all[i])
        fold1_label.append(temp)
    if S in f2:
        fold2_index.append(path_all[i])
        fold2_label.append(temp)
    if S in f3:
        fold3_index.append(path_all[i])
        fold3_label.append(temp)
    if S in f4:
        fold4_index.append(path_all[i])
        fold4_label.append(temp)
    if S in f5:
        fold5_index.append(path_all[i])
        fold5_label.append(temp)

fold1_label = np.array(fold1_label).astype(int)
fold2_label = np.array(fold2_label).astype(int)
fold3_label = np.array(fold3_label).astype(int)
fold4_label = np.array(fold4_label).astype(int)
fold5_label = np.array(fold5_label).astype(int)

def main(fold1,fold2,fold3,fold4,fold5,foldL1,foldL2,foldL3,foldL4,foldL5, Modelname, rightau, rightexp, leftexp):
    trainD_path = np.concatenate([fold1,fold2,fold3,fold4],axis=0)
    testD_path = fold5
    trainL = np.concatenate([foldL1,foldL2,foldL3,foldL4],axis=0)
    testL = foldL5

    pickle_in = open(os.path.join( '../BNwithConstraints/PseudoBN/'+'list_AU_config.p'), "rb" )
    load_AUconfig = pickle.load(pickle_in)
    
    pickle_in = open( os.path.join( '../BNwithConstraints/PseudoBN/Total/'+ 'PGM_p.p'), "rb" )
    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    
    print('finish loading BN')
    epo = 5
    out_epo = 5
    batch_size = 32
    
    perf_exp_training = []
    perf_exp_testing = []
    
    perf_au_training = []
    perf_au_testing = []
    
    perf_posterior = []
    
    total_samples = len(trainD_path)
    
    model_folder = os.path.join(os.path.join(os.getcwd()), 'Models')
    rightAU_model_path = os.path.join(model_folder, rightau)
    right_AU_model = ImportRightAU(rightAU_model_path)
    
    rightExp_model_path = os.path.join(model_folder, rightexp)
    right_Exp_model = ImportRightExp(rightExp_model_path)
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
        posterior_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 6], name='posterior')
        vgg = vgg19.Vgg19(os.path.join( './Models/'+ Modelname))
        vgg.build(images_norm, train_mode)
        print(vgg.get_var_count())

        sess.run(tf.global_variables_initializer())
        loss_exp = Loss_ExpressionModel_Labelonly(vgg.prob_exp, true_out)
        pred_Expression = tf.math.argmax(vgg.prob_exp, axis = 1)
        loss2 = Loss_ExpressionModel(vgg.fc_exp, true_out, posterior_, balance_e_)
        
        learn_rate=0.002
        train_vgg = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_exp)
        train_vgg2 = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss2)
    
    balance_e = 0.001 #for left expression
    balance_w =  0.001 #for right AU
    best_exp = -10
    best_au = -10
    best_au_arr = []
    print('start')
    for out_e in range(out_epo):
        start_idx_train = np.arange(0, len(trainL), batch_size)
        for ti in range(len(start_idx_train)): 
            train_image, train_explabel, train_AUlabel, train_AUprob, train_image_224 = get_train_batch(batch_size, trainD_path, trainL, load_PGM_pAUconfig, start_idx_train[ti])
            _, temp_pred = sess.run([loss_exp, vgg.prob_exp], feed_dict={images: train_image_224, true_out: train_explabel, train_mode: False})
            p_au, _, _ = right_AU_model.run(train_image, train_AUprob, load_AUconfig)
            _, _, p_exp_right, _, _ = right_Exp_model.run(p_au, train_explabel, load_PGM_pExp)
            temp_posterior_train = posterior_entropy(p_exp_right, temp_pred)
            if ti==0:
                posterior_train = temp_posterior_train
            else:
                posterior_train = np.concatenate((posterior_train, temp_posterior_train), axis=0)
        
        start_idx_test = np.arange(0, len(testL), batch_size)
        for ti in range(len(start_idx_test)): 
            test_image, test_explabel, test_AUlabel, test_AUprob, test_image_224 = get_train_batch(batch_size, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
            _, temp_pred = sess.run([loss_exp, vgg.prob_exp], feed_dict={images: test_image_224, true_out: test_explabel, train_mode: False})
            p_au, _, _ = right_AU_model.run(test_image, test_AUprob, load_AUconfig)
            _, _, p_exp_right, _, _ = right_Exp_model.run(p_au, test_explabel, load_PGM_pExp)
            temp_posterior_test = posterior_entropy(p_exp_right, temp_pred)
            if ti==0:
                posterior_test = temp_posterior_test
            else:
                posterior_test = np.concatenate((posterior_test, temp_posterior_test), axis=0)
                        
        acc_train = (np.ravel(trainL[:,0]-1)== np.argmax(posterior_train, axis=1)).sum()/len(trainL)
        acc_test = (np.ravel(testL[:,0]-1)== np.argmax(posterior_test, axis=1)).sum()/len(testL)
        
        perf_posterior.append([acc_train, acc_test])
        print('----------Outer_epo %d Posterior Acc Test(2) %f-------------' %(out_e, acc_test))
        print(' ')
        LR = 0.0001
        for e in range(epo): #each epo go over all samples
            
            '''performance on testing for each epo'''
            test_image, test_explabel, test_AUlabel, test_AUprob, test_image_224= get_train_batch(900, testD_path, testL, load_PGM_pAUconfig, start_idx_test[0])
            
            posterior = posterior_test
            
            p_au, _, _ = right_AU_model.run(test_image, test_AUprob, load_AUconfig)
            # left exp output
            loss_left, pred_exp_left = sess.run([loss2, pred_Expression], feed_dict={images: test_image_224, true_out: test_explabel, train_mode: False, posterior_:posterior, balance_e_: balance_e  })
            # right AU output
            _, pred_au, loss_au = right_AU_model.run_p2(test_image, test_AUprob, load_AUconfig, load_PGM_pAUconfig, posterior, balance_w)
            # right exp /PGM model
            _, pred_exp_K, _, pred_exp_right, loss_right = right_Exp_model.run(p_au, test_explabel, load_PGM_pExp)
            
            acc_left = (np.ravel(testL[:,0]-1)== pred_exp_left).sum()/len(testL)
            acc_right = (np.ravel(testL[:,0]-1)== pred_exp_right).sum()/len(testL)
            acc_K = (np.ravel(testL[:,0]-1)== pred_exp_K).sum()/len(testL)
            
            f1 = Compute_F1score_au(test_AUlabel, pred_au)
            perf_exp_testing.append([acc_left, acc_right,acc_K, loss_left, loss_right, loss_au])
            perf_au_testing.append(f1)

            if acc_left > best_exp:
                best_exp = acc_left
            if np.mean(f1) > best_au:
                best_au = np.mean(f1)
                best_au_arr = f1
            print('testing')
            print('e=%d, exp acc: left=%f, right=%f, au ave f1 = %f'%(e, acc_left, acc_right, np.mean(f1)))


            start_idx = np.arange(0, total_samples, batch_size)
            if e%10 == 0 and e != 0:
                LR = LR*0.9
            order=np.random.permutation(len(trainD_path))
            for i in range(len(start_idx)): 
                
                index = start_idx[i]
                train_image, train_explabel, train_AUlabel, train_AUprob,train_image_224 = get_train_batch(batch_size, trainD_path[order], trainL[order,:], load_PGM_pAUconfig, index)
                posterior_train2 = posterior_train[order,:]
                posterior = posterior_train2[index:(batch_size+index), :]
                
                p_au, _, _ = right_AU_model.run(train_image, train_AUprob, load_AUconfig)
                #train left exp
                #left_Exp_model.train_p2(train_image, train_explabel, posterior, balance_e, LR)

                loss_left,_, pred_exp_left = sess.run([loss2,train_vgg2, pred_Expression], feed_dict={images: train_image_224, true_out: train_explabel, train_mode: True, posterior_:posterior, balance_e_: balance_e  })
                #train right exp 
                right_Exp_model.train(p_au, train_explabel, LR, load_PGM_pExp)

                #train right AU
                right_AU_model.train_p2(train_image, train_AUprob, LR, load_AUconfig, load_PGM_pAUconfig, posterior, balance_w)
            

       
    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, leftexp), "wb" )
    pickle.dump([perf_exp_testing, perf_au_testing, best_exp, best_au, best_au_arr], pickle_out)
    
    pickle_out.close()
    
method = 'pseudoTotal'
aufilename_srting1 = 'AU_model_5fold_1mmi_'+method 
aufilename_srting2 = 'AU_model_5fold_2mmi_'+method 
aufilename_srting3 = 'AU_model_5fold_3mmi_'+method 
aufilename_srting4 = 'AU_model_5fold_4mmi_'+method 
aufilename_srting5 = 'AU_model_5fold_5mmi_'+method 

expfilename_srting1 = 'RightExp_5fold_1mmi_'+method 
expfilename_srting2 = 'RightExp_5fold_2mmi_'+method 
expfilename_srting3 = 'RightExp_5fold_3mmi_'+method 
expfilename_srting4 = 'RightExp_5fold_4mmi_'+method 
expfilename_srting5 = 'RightExp_5fold_5mmi_'+method 

Ffilename_srting1 = 'Final_5fold_1mmi_'+method 
Ffilename_srting2 = 'Final_5fold_2mmi_'+method 
Ffilename_srting3 = 'Final_5fold_3mmi_'+method 
Ffilename_srting4 = 'Final_5fold_4mmi_'+method 
Ffilename_srting5 = 'Final_5fold_5mmi_'+method

main(fold1_index,fold2_index,fold3_index,fold4_index,fold5_index,fold1_label,fold2_label,fold3_label,fold4_label,fold5_label,'LeftExp-fold1_pre_mmi.npy',aufilename_srting1,  expfilename_srting1, Ffilename_srting1)
main(fold5_index,fold1_index,fold2_index,fold3_index,fold4_index,fold5_label,fold1_label,fold2_label,fold3_label,fold4_label,'LeftExp-fold2_pre_mmi.npy',aufilename_srting2,  expfilename_srting2, Ffilename_srting2)
main(fold4_index,fold5_index,fold1_index,fold2_index,fold3_index,fold4_label,fold5_label,fold1_label,fold2_label,fold3_label,'LeftExp-fold3_pre_mmi.npy',aufilename_srting3,  expfilename_srting3, Ffilename_srting3)
main(fold3_index,fold4_index,fold5_index,fold1_index,fold2_index,fold3_label,fold4_label,fold5_label,fold1_label,fold2_label,'LeftExp-fold4_pre_mmi.npy',aufilename_srting4,  expfilename_srting4, Ffilename_srting4)
main(fold2_index,fold3_index,fold4_index,fold5_index,fold1_index,fold2_label,fold3_label,fold4_label,fold5_label,fold1_label,'LeftExp-fold5_pre_mmi.npy',aufilename_srting5,  expfilename_srting5, Ffilename_srting5)

pickleFolder = os.path.join( os.getcwd(), 'Results')
pickle_in = open( os.path.join( pickleFolder, Ffilename_srting1), "rb" )
_,_,bestexp1,bestau1, bestauarr1  = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, Ffilename_srting2), "rb" )
_,_,bestexp2,bestau2, bestauarr2 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, Ffilename_srting3), "rb" )
_,_,bestexp3,bestau3, bestauarr3 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, Ffilename_srting4), "rb" )
_,_,bestexp4,bestau4, bestauarr4 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, Ffilename_srting5), "rb" )
_,_,bestexp5,bestau5, bestauarr5 = pickle.load(pickle_in)


print('MMI:%s'%(method))
print('AU 1=%f, 2=%f, 3=%f, 4=%f,5=%f, ave=%f'%(bestau1, bestau2, bestau3, bestau4, bestau5, (bestau1+bestau2+bestau3+bestau4+bestau5)/5))
print((bestauarr1+bestauarr2+bestauarr3+bestauarr4+bestauarr5)/5)
print('FER 1=%f, 2=%f, 3=%f, 4=%f,5=%f, ave=%f'%(bestexp1,bestexp2, bestexp3, bestexp4, bestexp5, (bestexp1+bestau2+bestexp3+bestexp4+bestexp5)/5))


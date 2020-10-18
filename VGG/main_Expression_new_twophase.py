# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:56:11 2019

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
from helper_functions_new import weight_variable, bias_variable, conv2d, max_pool_2x2, get_train_batch, get_valid_test_set
from helper_functions import Compute_F1score, Compute_F1score_au,get_train_batch_ck
from Expression_part_new import CNN_Expression, Loss_ExpressionModel, Loss_ExpressionModel_Labelonly
from Update_Posterior import Exp_KnowledgeModel
from ImportGraph_twophase import ImportRightAU, ImportRightExp, ImportLeftExp
#from Ipython.display import Image, display
import pdb
import itertools
import scipy.io as sio

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #(e.g. it will use GPU’s number 0 and 2.)

#training, testing index
pickleFolder = os.path.join( os.getcwd(), 'data_pickle') 
#pickle_in = open( os.path.join( pickleFolder, '3fold-image-serverpath.p'), "rb" )

#fold1_path, fold2_path, fold3_path = pickle.load(pickle_in)
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

# for PGM
# PGM_data = np.concatenate((label_exp, label_AU ), axis = 1)
# sio.savemat('data.mat',{'data':PGM_data})
# pdb.set_trace()
fold1_index=path_all[0:248]
fold2_index=path_all[248:492]
fold3_index=path_all[492:732]
fold1_label=np.concatenate((label_exp[0:248], label_AU[0:248, :] ), axis = 1) #5AU index [0,1,5,6,9]
fold2_label=np.concatenate((label_exp[248:492], label_AU[248:492, :] ), axis = 1)
fold3_label=np.concatenate((label_exp[492:732], label_AU[492:732, :] ), axis = 1)

fold_BP=path_all
fold_BPlabel=np.concatenate((label_exp, label_AU), axis = 1)
# CK+
data=sio.loadmat('/home/zijun/Documents/CK+/'+'CK+_6_BASIC.mat')
path=data['path']
label=data['labels']
for i in range(label.shape[0]):
    if label[i,0]==1:
        label[i,0]=label[i,0]+1
#training, testing index
fold1_in=path[0:64]
fold2_in=path[64:137]
fold3_in=path[137:193]
fold4_in=path[193:243]
fold5_in=path[244:309]
fold1_la=label[0:64,:]
fold2_la=label[64:137,:]
fold3_la=label[137:193,:]
fold4_la=label[193:243,:]
fold5_la=label[244:309,:]

fold1_index=path[0:277]
fold2_index=path[277:309]
fold1_label=label[0:277,:]
fold2_label=label[277:309,:]
train_list=[]
train_la=np.zeros([257,9],dtype=int)
test_list=[]
test_la=np.zeros([24,9],dtype=int)
t=0
for i in range(len(fold1_index)):
    if fold1_label[i,0]!=6:
        train_list.append(fold1_index[i])
        train_la[t,:]=fold1_label[i,:]
        if fold1_label[i,0]==7:
            train_la[t,0]=6
        t=t+1
t=0
for i in range(len(fold2_index)):
    if fold2_label[i,0]!=6:
        test_list.append(fold2_index[i])
        test_la[t,:]=fold2_label[i,:]
        if fold2_label[i,0]==7:
            test_la[t,0]=6 
        t=t+1

fold_CK = train_list+test_list
fold_CKlabel= np.concatenate((train_la, test_la), axis = 0)


def main(fold_BP, fold_BPlabel, fold_CK, fold_CKlabel, title_string):
    # trainD_path = list(itertools.chain(train1_index, train2_index))
    # testD_path = test_index
    
    # trainL = np.concatenate((train1_label, train2_label), axis=0)
    # testL = test_label

    trainD_path = fold_CK
    testD_path = fold_BP
    
    trainL = fold_CKlabel
    testL = fold_BPlabel
    
    
    pickle_in = open(os.path.join( '/home/zijun/Documents/CK+/PGMmodels/BP4D-to-CK-8AU/K2_Bayes/'+'list_AU_config.p'), "rb" )
    load_AUconfig = pickle.load(pickle_in)
    #pickle_in = open( os.path.join( '/home/zijun/Documents/CK+/PGMmodels/pseudoBN/6exp-8AU/'+ 'PGM_p-K2-Bayes.p'), "rb" )
    
    pickle_in = open( os.path.join( '/home/zijun/Documents/CK+/PGMmodels/BP4D-to-CK-8AU/K2_Bayes/'+ 'PGM_p-K2-Bayes.p'), "rb" )
    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    

    batch_size = 50
    epo = 40
    LR = 0.0005
    
    #expression model performance [loss, f1score]
    perf_exp_training = []
    
    perf_exp_testing = []


    total_samples = len(trainD_path)
    
    define_leftexp_file_name = 'define_LeftExp_p2'
    model_folder = os.path.join(os.path.join(os.getcwd()), 'Models')
    define_leftexp_path = os.path.join(model_folder, define_leftexp_file_name)

    left_Exp_model = ImportLeftExp(define_leftexp_path)
    
    for e in range(epo): #each epo go over all samples
            
        #training iteration
        start_idx = np.arange(0, total_samples, batch_size)
        for i in range(len(start_idx)): 
            
            index = start_idx[i]
            #train_image, train_explabel,au ,train_pro = get_train_batch_ck(batch_size, trainD_path, trainL, load_PGM_pAUconfig, index)
            train_image, train_explabel = get_train_batch(batch_size, trainD_path, trainL, load_PGM_pAUconfig, index)
            left_Exp_model.train(train_image, train_explabel, LR)


        # start_idx_train = np.arange(0, len(trainL), 300)
        # for ti in range(len(start_idx_train)): 
        #     train_image, train_explabel = get_train_batch(300, trainD_path, trainL, load_PGM_pAUconfig, start_idx_train[ti])
            
        #     _, temp_pred, temp_loss = left_Exp_model.run(train_image, train_explabel) #p_exp, pred_exp, loss

        #     if ti==0:
        #         train_loss_b = np.expand_dims(temp_loss, axis=0)
        #         train_pred_b = temp_pred
        #     else:
        #         train_loss_b = np.concatenate((train_loss_b, np.expand_dims(temp_loss, axis=0)), axis=0)
        #         train_pred_b = np.concatenate((train_pred_b, temp_pred), axis=0)
        # np_train_loss_exp = np.mean(train_loss_b)
        # np_acc_train_exp =  (np.ravel(trainL[:,0]-1)== train_pred_b).sum()/len(trainL)
        # np_f1_train_exp = 0
        # perf_exp_training.append([np_train_loss_exp, np_acc_train_exp, np_f1_train_exp])
        # print("epoch:%d,training ACC:%f,F1:%f,"%(e,np_acc_train_exp,np_f1_train_exp,))
        # print(' ')

    
        start_idx_test = np.arange(0, len(testL), 100)
        for ti in range(len(start_idx_test)): 
            #test_image, test_explabel = get_train_batch(100, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
            
            test_image, test_explabel,_,_ = get_train_batch_ck(100, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
            
            _, temp_pred, temp_loss = left_Exp_model.run(test_image, test_explabel) #p_exp, pred_exp, loss

            if ti==0:
                test_loss_b = np.expand_dims(temp_loss, axis=0)
                test_pred_b = temp_pred
            else:
                test_loss_b = np.concatenate((test_loss_b, np.expand_dims(temp_loss, axis=0)), axis=0)
                test_pred_b = np.concatenate((test_pred_b, temp_pred), axis=0)
        np_test_loss_exp = np.mean(test_loss_b)
        np_acc_test_exp =  (np.ravel(testL[:,0]-2)== test_pred_b).sum()/len(testL)
        np_f1_test_exp = 0
        perf_exp_testing.append([np_test_loss_exp, np_acc_test_exp, np_f1_test_exp])
        print("epoch:%d,testing ACC:%f,F1:%f,"%(e,np_acc_test_exp,np_f1_test_exp))
        print(' ')

    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, title_string), "wb" )
    pickle.dump([perf_exp_training, perf_exp_testing], pickle_out)
    
    pickle_out.close()
    
    file_name = title_string[:-2]
    write_model_path = os.path.join(model_folder, file_name)
    left_Exp_model.save(define_leftexp_path, write_model_path)
    

name_string = 'initialLeftExp-joint'
file1 = [name_string + '-fold1.p'][0]
file2 = [name_string + '-fold2.p'][0]
file3 = [name_string + '-fold3.p'][0]

file0='initialLeftExp-joint_cross1.p'
#main(fold_BP, fold_BPlabel, fold_CK, fold_CKlabel, file0)
main(fold_CK, fold_CKlabel, fold_BP, fold_BPlabel, file0)
pdb.set_trace()
main(fold1_index, fold2_index, fold3_index, fold1_label, fold2_label, fold3_label, file1)
main(fold1_index, fold3_index, fold2_index, fold1_label, fold3_label, fold2_label, file2)
main(fold2_index, fold3_index, fold1_index, fold2_label, fold3_label, fold1_label, file3)

pickleFolder = os.path.join( os.getcwd(), 'Results') 
pickle_in = open( os.path.join( pickleFolder, file1), "rb" )
perf_train_1, perf_test_1 = pickle.load(pickle_in)

pickle_in = open( os.path.join( pickleFolder, file2), "rb" )
perf_train_2, perf_test_2 = pickle.load(pickle_in)

pickle_in = open( os.path.join( pickleFolder, file3), "rb" )
perf_train_3, perf_test_3 = pickle.load(pickle_in)


idx1 = np.argmax(np.asarray(perf_test_1)[:,1])
idx2 = np.argmax(np.asarray(perf_test_2)[:,1])
idx3 = np.argmax(np.asarray(perf_test_3)[:,1])

perf_train = [perf_train_1[idx1], perf_train_2[idx2], perf_train_3[idx3]]
perf_test = [perf_test_1[idx1], perf_test_2[idx2], perf_test_3[idx3]]

acc_train = np.mean(np.asarray(perf_train)[:,1])
f1_train = np.mean(np.asarray(perf_train)[:,2])
acc = np.mean(np.asarray(perf_test)[:,1])
f1 = np.mean(np.asarray(perf_test)[:,2])


print("left expression model training ACC:%f,F1:%f"%(acc_train, f1_train))
print(' ')
print("left expression model testing ACC:%f,F1:%f"%(acc, f1))
print(' ')



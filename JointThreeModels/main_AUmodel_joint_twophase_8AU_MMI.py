import numpy as np
import tensorflow as tf
import matplotlib.image as mping
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import f1_score
from scipy.io import loadmat
from os.path import isfile, join
from helper_functions import weight_variable, bias_variable, conv2d, max_pool_2x2, get_train_batch, get_valid_test_set
from helper_functions import Compute_F1score, Compute_F1score_au,get_train_batch_pro,get_train_batch_ck
from AU_Knowledge_part_new import CNN_AUdetection_joint, Loss_KnowledgeModel_joint, Loss_KnowledgeModel_gtExpOnly_joint
from Update_Posterior import Exp_KnowledgeModel_joint
from ThirdModel_part import Loss_3rdModel, Prediction_3rdModel_joint
from ImportGraph_twophase import ImportRightAU, ImportRightExp
#from Ipython.display import Image, display
import itertools
import scipy.io as sio
import cv2
import pdb
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  #(e.g. it will use GPU’s number 0 and 2.)

pickleFolder = os.path.join( os.getcwd()) 
data = sio.loadmat(os.path.join(pickleFolder, 'MMI_8AU_6Exp.mat'))
index = data['MMI_8AU_6Exp']
sub_name = index[0,0]['SUB']
session_name = index[0,0]['Session']
frame_name = index[0,0]['Frame']
label_exp = index[0,0]['EXP']
label_AU = index[0,0]['AU']
path_all=[]
for i in range(sub_name.shape[0]):
    path = ["/home/zijun/Documents/MMI/RawFrames_peak_crop/%d-"%(session_name[i])+"%d"%(frame_name[i])+".jpg"]
    if i == 0:
        path_all=path
    else:
        path_all=path_all + path
#label_ = sio.loadmat('BP4D_8AU.mat')
#label_AU = label_['AU_8']
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

#fold1_index=path_all[0:99]
#fold2_index=path_all[99:201]
#fold3_index=path_all[201:300]
#fold4_index=path_all[300:399]
#fold5_index=path_all[399:504]
#fold1_label=np.concatenate((label_exp[0:99], label_AU[0:99, :] ), axis = 1) #5AU index [0,1,5,6,9]
#fold2_label=np.concatenate((label_exp[99:201], label_AU[99:201, :] ), axis = 1)
#fold3_label=np.concatenate((label_exp[201:300], label_AU[201:300, :] ), axis = 1)
#fold4_label=np.concatenate((label_exp[300:399], label_AU[300:399, :] ), axis = 1)
#fold5_label=np.concatenate((label_exp[399:504], label_AU[399:504, :] ), axis = 1)

#fold1_sub=sub_name[0:165]
#fold2_sub=sub_name[165:357]
#fold3_sub=sub_name[357:504]
#fold4_sub=sub_name[357:504]
#fold5_sub=sub_name[357:504]

def main(fold1,fold2,fold3,fold4,fold5,foldL1,foldL2,foldL3,foldL4,foldL5, title_string):
    
    trainD_path = np.concatenate([fold1,fold2,fold3,fold4],axis=0)
    testD_path = fold5
    trainL = np.concatenate([foldL1,foldL2,foldL3,foldL4],axis=0)
    testL = foldL5

    pickle_in = open(os.path.join( '/home/zijun/Documents/PseudoBN/'+'list_AU_config.p'), "rb" )
    load_AUconfig = pickle.load(pickle_in)
    
#    pickle_in = open( os.path.join( '/home/zijun/Documents/BNwGTlabel/MMI/K2-Bayes/'+ 'PGM_p.p'), "rb" )
#    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    
    #PGM_p_Total/ PGM_p_ExpDepJoint / PGM_p_ExpDepSingle / PGM_p_ExpIndepJoint
    pickle_in = open( os.path.join( '/home/zijun/Documents/PseudoBN/Total/'+ 'PGM_p_1.p'), "rb" )
    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    
    batch_size = 16
    epo = 50
    LR = 0.0003
    
    #AU model performance [total_loss, f1score for each AU]
    perf_au_training = []
    perf_au_testing = []
    
    total_samples = len(trainD_path)
    
    file_name = 'define_AU_p2_8AU'
    model_folder = os.path.join(os.path.join(os.getcwd()), 'Models')
    define_model_path = os.path.join(model_folder, file_name)
    #    model_path = os.path.join('E:/Models', file_name)
    right_AU_model = ImportRightAU(define_model_path)
    
    best = -10
    for e in range(epo): #each epo go over all samples
        print('--------------epo: %d --------------' %e)
        if e%20 == 0 and e!= 0:
            LR = LR*0.9
        #training iteration
        # shuffle
        rand_idx = np.random.permutation(total_samples)
        start_idx = np.arange(0, total_samples, batch_size)
        for i in range(len(start_idx)): 
            index = start_idx[i]
            train_image ,_, train_AUlabel, train_pro,_= get_train_batch(batch_size, trainD_path[rand_idx], trainL[rand_idx,:], load_PGM_pAUconfig, index)            
            right_AU_model.train(train_image, train_pro, LR, load_AUconfig) #image, AUprob, learning_rate, AUconfig
            
        #training for each epo
#        start_idx_train = np.arange(0, len(trainD_path), 300)
#        for ti in range(len(start_idx_train)): 
#            train_image, _, train_AUlabel, train_pro,_= get_train_batch(300, trainD_path, trainL, load_PGM_pAUconfig, start_idx_train[ti])
#            
#            _, temp_pred_au, temp_loss_au = right_AU_model.run(train_image, train_pro, load_AUconfig) #image, AUprob, AUconfig
#
#            if ti==0:
#                train_loss_au_b = np.expand_dims(temp_loss_au, axis=0)
#                train_pred_au_b = temp_pred_au
#            else:
#                train_loss_au_b = np.concatenate((train_loss_au_b, np.expand_dims(temp_loss_au, axis=0)), axis=0)
#                train_pred_au_b = np.concatenate((train_pred_au_b, temp_pred_au), axis=0)
#        
#        #np_f1_train_au = Compute_F1score_au(trainL[:,1:], train_pred_au_b)
#        
#        np_f1_train_au = Compute_F1score_au(trainL[:,1:], train_pred_au_b)
#        perf_au_training.append([np.mean(train_loss_au_b), np_f1_train_au])
#        print('training')
#        print('AU total loss || ave f1')
#        print(np.mean(train_loss_au_b), np.mean(np_f1_train_au))
#        print(' ')
        
        #testing for each epo
        start_idx_test = np.arange(0, len(testD_path), 300)
        for ti in range(len(start_idx_test)):
            test_image, test_explabel, test_AUlabel, test_AUprob,_ = get_train_batch(300, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
            
            _, temp_pred_au, temp_loss_au = right_AU_model.run(test_image, test_AUprob, load_AUconfig) #image, AUprob, AUconfig            
    
            if ti==0:
                test_loss_au_b = np.expand_dims(temp_loss_au, axis=0)
                test_pred_au_b = temp_pred_au
            else:
                test_loss_au_b = np.concatenate((test_loss_au_b, np.expand_dims(temp_loss_au, axis=0)), axis=0)
                test_pred_au_b = np.concatenate((test_pred_au_b, temp_pred_au), axis=0)
        
        
        np_f1_test_au = Compute_F1score_au(testL[:,1:], test_pred_au_b)
        perf_au_testing.append([np.mean(test_loss_au_b), np_f1_test_au])
        
        print('testing')
        print('AU total loss || ave f1')
        print(np.mean(test_loss_au_b), np.mean(np_f1_test_au))
        print(' ')
        if np.mean(np_f1_test_au) > best:
            best = np.mean(np_f1_test_au)
            best_arr = np_f1_test_au
            file_name = title_string
            write_model_path = os.path.join(model_folder, file_name)
            right_AU_model.save(define_model_path, write_model_path)
    
    print('best')
    print(best) 
    
    pickleFolder = os.path.join(os.getcwd(),'Results')
    result_folder = os.path.join(pickleFolder)
    
    pickle_out = open( os.path.join( result_folder, title_string), "wb" )
    pickle.dump([perf_au_training, perf_au_testing, best, best_arr], pickle_out)
    
    pickle_out.close()
    
#    file_name = title_string
#    write_model_path = os.path.join(model_folder, file_name)
#    right_AU_model.save(define_model_path, write_model_path)

method = 'pseudoTotal'
filename_srting1 = 'AU_model_5fold_1mmi_'+method 
filename_srting2 = 'AU_model_5fold_2mmi_'+method 
filename_srting3 = 'AU_model_5fold_3mmi_'+method 
filename_srting4 = 'AU_model_5fold_4mmi_'+method 
filename_srting5 = 'AU_model_5fold_5mmi_'+method 

#main(fold1_index,fold2_index,fold3_index,fold4_index,fold5_index,fold1_label,fold2_label,fold3_label,fold4_label,fold5_label, filename_srting1)
#main(fold2_index,fold3_index,fold4_index,fold5_index,fold1_index,fold2_label,fold3_label,fold4_label,fold5_label,fold1_label, filename_srting2)
#main(fold3_index,fold4_index,fold5_index,fold1_index,fold2_index,fold3_label,fold4_label,fold5_label,fold1_label,fold2_label, filename_srting3)
main(fold4_index,fold5_index,fold1_index,fold2_index,fold3_index,fold4_label,fold5_label,fold1_label,fold2_label,fold3_label, filename_srting4)
#main(fold5_index,fold1_index,fold2_index,fold3_index,fold4_index,fold5_label,fold1_label,fold2_label,fold3_label,fold4_label, filename_srting5)


pickleFolder = os.path.join( os.getcwd(), 'Results') 

pickle_in = open( os.path.join( pickleFolder, filename_srting1), "rb" )
_, _,best1,bestarr1 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, filename_srting2), "rb" )
_, _, best2,bestarr2 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, filename_srting3), "rb" )
_, _, best3,bestarr3 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, filename_srting4), "rb" )
_, _, best4,bestarr4 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, filename_srting5), "rb" )
_, _, best5,bestarr5 = pickle.load(pickle_in)

print('mmi: %s'%(method))
print('1=%f, 2=%f, 3=%f, 4=%f,5=%f, ave=%f'%(best1, best2, best3, best4, best5, (best1+best2+best3+best4+best5)/5))
print((bestarr1+bestarr2+bestarr3+bestarr4+bestarr5)/5)
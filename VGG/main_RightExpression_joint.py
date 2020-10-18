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
from helper_functions import Compute_F1score, Compute_F1score_au,get_train_batch_ck
from AU_Knowledge_part_new import CNN_AUdetection_joint, Loss_KnowledgeModel_joint, Loss_KnowledgeModel_gtExpOnly_joint
from Update_Posterior import posterior_entropy, posterior_summation, Exp_KnowledgeModel_joint
from ThirdModel_part import Loss_3rdModel, Prediction_3rdModel_joint
from ImportGraph import ImportRightAU, ImportRightExp_BP4D
#from Ipython.display import Image, display
import itertools
import scipy.io as sio
import pdb
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
fold1_index=path_all[0:248]
fold2_index=path_all[248:492]
fold3_index=path_all[492:732]
fold1_label=np.concatenate((label_exp[0:248], label_AU[0:248, :] ), axis = 1) #5AU index [0,1,5,6,9]
fold2_label=np.concatenate((label_exp[248:492], label_AU[248:492, :] ), axis = 1)
fold3_label=np.concatenate((label_exp[492:732], label_AU[492:732, :] ), axis = 1)


#fold_BP=path_all
#fold_BPlabel=np.concatenate((label_exp, label_AU), axis = 1)
## CK+
#data=sio.loadmat('/home/zijun/Documents/CK+/'+'CK+_6_BASIC.mat')
#path=data['path']
#label=data['labels']
#for i in range(label.shape[0]):
#    if label[i,0]==1:
#        label[i,0]=label[i,0]+1
##training, testing index
#fold1_in=path[0:64]
#fold2_in=path[64:137]
#fold3_in=path[137:193]
#fold4_in=path[193:243]
#fold5_in=path[244:309]
#fold1_la=label[0:64,:]
#fold2_la=label[64:137,:]
#fold3_la=label[137:193,:]
#fold4_la=label[193:243,:]
#fold5_la=label[244:309,:]
#
#fold1_index=path[0:277]
#fold2_index=path[277:309]
#fold1_label=label[0:277,:]
#fold2_label=label[277:309,:]
#train_list=[]
#train_la=np.zeros([257,9],dtype=int)
#test_list=[]
#test_la=np.zeros([24,9],dtype=int)
#t=0
#for i in range(len(fold1_index)):
#    if fold1_label[i,0]!=6:
#        train_list.append(fold1_index[i])
#        train_la[t,:]=fold1_label[i,:]
#        if fold1_label[i,0]==7:
#            train_la[t,0]=6
#        t=t+1
#t=0
#for i in range(len(fold2_index)):
#    if fold2_label[i,0]!=6:
#        test_list.append(fold2_index[i])
#        test_la[t,:]=fold2_label[i,:]
#        if fold2_label[i,0]==7:
#            test_la[t,0]=6 
#        t=t+1
#
#fold_CK = train_list+test_list
#fold_CKlabel= np.concatenate((train_la, test_la), axis = 0)


def main(train1_index, train2_index, test_index, train1_label, train2_label, test_label, file_au, file_):
    trainD_path = list(itertools.chain(train1_index, train2_index))
    testD_path = test_index
    
    trainL = np.concatenate((train1_label, train2_label), axis=0)
    testL = test_label

#    trainD_path = fold_CK
#    testD_path = fold_BP
#    
#    trainL = fold_CKlabel
#    testL = fold_BPlabel
    #
    
    # pickle_in = open(os.path.join( '/home/zijun/Documents/CK+/PGMmodels/BP4D-to-CK-8AU/K2_Bayes/'+'list_AU_config.p'), "rb" )
    # load_AUconfig = pickle.load(pickle_in)
    # #pickle_in = open( os.path.join( '/home/zijun/Documents/CK+/PGMmodels/pseudoBN/6exp-8AU/'+ 'PGM_p-K2-Bayes.p'), "rb" )
    
    # pickle_in = open( os.path.join( '/home/zijun/Documents/CK+/PGMmodels/BP4D-to-CK-8AU/K2_Bayes/'+ 'PGM_p-K2-Bayes.p'), "rb" )
    # load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    pickle_in = open( '/home/zijun/Documents/BP4D/data_pickle/BP4D-8AU-5Exp/list_AU_config.p', "rb" )
    load_AUconfig = pickle.load(pickle_in)
    
    pickle_in = open( '/home/zijun/Documents/BP4D/data_pickle/BP4D-8AU-5Exp/PGM_p-K2-Bayes.p', "rb" )
    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    
    batch_size = 50
    epo = 200
    LR = 0.0001
    
    perf_3rd_training = []
    perf_3rd_testing = []
    
    perf_K_training = []
    perf_K_testing = []
    
    total_samples = len(trainD_path)
    
    define_rightexp_file_name = 'define_right_exp'
    model_folder = os.path.join(os.path.join(os.getcwd()), 'Models')
    define_rightexp_path = os.path.join(model_folder, define_rightexp_file_name)
    
    AU_model_name = file_au[:-2]
    AUmodel_path = os.path.join(model_folder, AU_model_name)
    right_AU_model = ImportRightAU(AUmodel_path)
    
    right_Exp_model = ImportRightExp_BP4D(define_rightexp_path)
    best = -10
    for e in range(epo): #each epo go over all samples
        print('--------------epo--------------%d' %e)
        #training iteration
        start_idx = np.arange(0, total_samples, batch_size)
        for i in range(len(start_idx)): 
            
            index = start_idx[i]
            train_image, train_explabel, train_AUlabel, train_AUprob = get_train_batch(batch_size, trainD_path, trainL, load_PGM_pAUconfig, index)
            
            np_p_au, _, _ = right_AU_model.run(train_image, train_AUprob, load_AUconfig) 
            
            right_Exp_model.train(np_p_au, train_explabel, LR, load_PGM_pExp)
            
            
#        #training for each epo
#        start_idx_train = np.arange(0, len(trainD_path), 300)
#        for ti in range(len(start_idx_train)): 
#            train_image, train_explabel, train_AUlabel, train_AUprob = get_train_batch(300, trainD_path, trainL, load_PGM_pAUconfig, start_idx_train[ti])
#            
#            np_p_au, _, _ = right_AU_model.run(train_image, train_AUprob, load_AUconfig)
#            
#            _, temp_pred_K, _, temp_pred_3rd, temp_loss_3rd = right_Exp_model.run(np_p_au, train_explabel, load_PGM_pExp)
#            
#            if ti==0:
#                train_loss_3rd_b = np.expand_dims(temp_loss_3rd, axis=0)
#                train_pred_3rd_b = temp_pred_3rd
#                train_pred_K_b = temp_pred_K
#            else:
#                train_loss_3rd_b = np.concatenate((train_loss_3rd_b, np.expand_dims(temp_loss_3rd, axis=0)), axis=0)
#                train_pred_3rd_b = np.concatenate((train_pred_3rd_b, temp_pred_3rd), axis=0)
#                train_pred_K_b = np.concatenate((train_pred_K_b, temp_pred_K), axis=0)
#    
#        np_acc_train_exp_3rd =  (np.ravel(trainL[:,0]-1)== train_pred_3rd_b).sum()/len(trainL)
#        np_f1_train_exp_3rd = 0
#        np_acc_train_exp_K = (np.ravel(trainL[:,0]-1)== train_pred_K_b).sum()/len(trainL)
#        np_f1_train_exp_K = 0
#        perf_3rd_training.append([np.mean(train_loss_3rd_b), np_acc_train_exp_3rd, np_f1_train_exp_3rd])
#        perf_K_training.append([np_acc_train_exp_K, np_f1_train_exp_K])
        
#        print('training')
#        print('3rd model loss || acc || f1')
#        print(np.mean(train_loss_3rd_b), np_acc_train_exp_3rd, np_f1_train_exp_3rd)
#        print(' ')
#        print('PGM model acc || f1')
#        print(np_acc_train_exp_K, np_f1_train_exp_K)
#        print(' ')
        
        #testing for each epo
        start_idx_test = np.arange(0, len(testD_path), 300)
        for ti in range(len(start_idx_test)): 
            test_image, test_explabel, test_AUlabel, test_AUprob = get_train_batch(300, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
            
            np_p_au, _, _ = right_AU_model.run(test_image, test_AUprob, load_AUconfig)
            
            _, temp_pred_K, _, temp_pred_3rd, temp_loss_3rd = right_Exp_model.run(np_p_au, test_explabel, load_PGM_pExp)
            
            if ti==0:
                test_loss_3rd_b = np.expand_dims(temp_loss_3rd, axis=0)
                test_pred_3rd_b = temp_pred_3rd
                test_pred_K_b = temp_pred_K
            else:
                test_loss_3rd_b = np.concatenate((test_loss_3rd_b, np.expand_dims(temp_loss_3rd, axis=0)), axis=0)
                test_pred_3rd_b = np.concatenate((test_pred_3rd_b, temp_pred_3rd), axis=0)
                test_pred_K_b = np.concatenate((test_pred_K_b, temp_pred_K), axis=0)
        
        np_acc_test_exp_3rd =  (np.ravel(testL[:,0]-1)== test_pred_3rd_b).sum()/len(testL)
        np_f1_test_exp_3rd = 0
        np_acc_test_exp_K = (np.ravel(testL[:,0]-1)== test_pred_K_b).sum()/len(testL)
        np_f1_test_exp_K = 0
        perf_3rd_testing.append([np.mean(test_loss_3rd_b), np_acc_test_exp_3rd, np_f1_test_exp_3rd])
        perf_K_testing.append([np_acc_test_exp_K, np_f1_test_exp_K])
        
#        print('testing')
#        print('3rd model acc')
#        print(np_acc_test_exp_3rd)
#        print(' ')
#        print('PGM model acc')
#        print(np_acc_test_exp_K)
#        print(' ')
        
        if np_acc_test_exp_3rd > best:
            best = np_acc_test_exp_3rd
            print('best')
            print(best)
            file_name = file_[:-2]
            write_model_path = os.path.join(model_folder, file_name)
            right_Exp_model.save(define_rightexp_path, write_model_path)
    
    
    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, file_), "wb" )
    pickle.dump([perf_3rd_training, perf_3rd_testing, perf_K_training, perf_K_testing], pickle_out)
    
    pickle_out.close()
    
#    file_name = file_[:-2]
#    write_model_path = os.path.join(model_folder, file_name)
#    right_Exp_model.save(define_rightexp_path, write_model_path)
    right_Exp_model.close()

name_string_au = 'initialAU-joint'
file1_au = [name_string_au + '-fold1-new.p'][0]
file2_au = [name_string_au + '-fold2-new.p'][0]
file3_au = [name_string_au + '-fold3-new.p'][0]

name_string = 'initialRightExp-joint'
file1 = [name_string + '-fold1-new.p'][0]
file2 = [name_string + '-fold2-new.p'][0]
file3 = [name_string + '-fold3-new.p'][0]


#file0='initialAU-joint_cross1.p'
#file1='initialRightExp-joint_cross1.p'
#main(fold_CK, fold_CKlabel, fold_BP, fold_BPlabel, file0, file1)
#pdb.set_trace()

main(fold1_index, fold2_index, fold3_index, fold1_label, fold2_label, fold3_label, file1_au, file1)
main(fold1_index, fold3_index, fold2_index, fold1_label, fold3_label, fold2_label, file2_au, file2)
main(fold2_index, fold3_index, fold1_index, fold2_label, fold3_label, fold1_label, file3_au, file3)

#pickleFolder = os.path.join( os.getcwd(), 'Results') 
#pickle_in = open( os.path.join( pickleFolder, file1), "rb" )
#perf_train_3rd1, perf_test_3rd1, perf_train_K1, perf_test_K1 = pickle.load(pickle_in)
#
#pickle_in = open( os.path.join( pickleFolder, file2), "rb" )
#perf_train_3rd2, perf_test_3rd2, perf_train_K2, perf_test_K2 = pickle.load(pickle_in)
#
#pickle_in = open( os.path.join( pickleFolder, file3), "rb" )
#perf_train_3rd3, perf_test_3rd3, perf_train_K3, perf_test_K3 = pickle.load(pickle_in)
#
#
#idx1 = np.argmax(np.asarray(perf_test_3rd1)[:,1])
#idx2 = np.argmax(np.asarray(perf_test_3rd2)[:,1])
#idx3 = np.argmax(np.asarray(perf_test_3rd3)[:,1])
#
#perf_train_3rd = [perf_train_3rd1[idx1], perf_train_3rd2[idx2], perf_train_3rd3[idx3]]
#perf_test_3rd = [perf_test_3rd1[idx1], perf_test_3rd2[idx2], perf_test_3rd3[idx3]]
#
#perf_train_K = [perf_train_K1[-1], perf_train_K2[-1], perf_train_K3[-1]]
#perf_test_K = [perf_test_K1[-1], perf_test_K2[-1], perf_test_K3[-1]]
#
#acc_3rd_train = np.mean(np.asarray(perf_train_3rd)[:,1])
#f1_3rd_train = np.mean(np.asarray(perf_train_3rd)[:,2])
#acc_3rd = np.mean(np.asarray(perf_test_3rd)[:,1])
#f1_3rd = np.mean(np.asarray(perf_test_3rd)[:,2])
#
#acc_K_train = np.mean(np.asarray(perf_train_K)[:,0])
#f1_K_train = np.mean(np.asarray(perf_train_K)[:,1])
#acc_K = np.mean(np.asarray(perf_test_K)[:,0])
#f1_K = np.mean(np.asarray(perf_test_K)[:,1])
#
#print("expression 3rd model training ACC:%f,F1:%f"%(acc_3rd_train, f1_3rd_train))
#print(' ')
#print("expression 3rd model testing ACC:%f,F1:%f"%(acc_3rd, f1_3rd))
#print(' ')
#print('expression PGM model training ACC:%f,F1:%f'%(acc_K_train, f1_K_train))
#print(' ')
#print('expression PGM model testing ACC:%f,F1:%f'%(acc_K, f1_K))
#

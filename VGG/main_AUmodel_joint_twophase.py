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
from helper_functions import Compute_F1score, Compute_F1score_au,get_train_batch_pro
from AU_Knowledge_part_new import CNN_AUdetection_joint, Loss_KnowledgeModel_joint, Loss_KnowledgeModel_gtExpOnly_joint
from Update_Posterior import Exp_KnowledgeModel_joint
from ThirdModel_part import Loss_3rdModel, Prediction_3rdModel_joint
from ImportGraph_twophase import ImportRightAU, ImportRightExp
#from Ipython.display import Image, display
import itertools
import scipy.io as sio
import pdb
import cv2
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

def main(train1_index, train2_index, test_index, train1_label, train2_label, test_label, title_string):
    trainD_path = list(itertools.chain(train1_index, train2_index))
    testD_path = test_index
    
    trainL = np.concatenate((train1_label, train2_label), axis=0)
    testL = test_label
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
    epo = 100
    LR = 0.0005
    
    #AU model performance [total_loss, f1score for each AU]
    perf_au_training = []
    perf_au_testing = []
    
    total_samples = len(trainD_path)
    
    file_name = 'define_AU_p2'
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
        start_idx = np.arange(0, total_samples, batch_size)
        for i in range(len(start_idx)): 
            index = start_idx[i]
            train_image ,_, train_AUlabel, train_pro= get_train_batch(batch_size, trainD_path, trainL, load_PGM_pAUconfig, index)
            
            right_AU_model.train(train_image, train_pro, LR, load_AUconfig) #image, AUprob, learning_rate, AUconfig
            
#        #training for each epo
#        start_idx_train = np.arange(0, len(trainD_path), 300)
#        for ti in range(len(start_idx_train)): 
#            train_image, _, train_AUlabel, train_pro= get_train_batch(300, trainD_path, trainL, load_PGM_pAUconfig, start_idx_train[ti])
#            
#            _, temp_pred_au, temp_loss_au = right_AU_model.run(train_image, train_pro, load_AUconfig) #image, AUprob, AUconfig
#            
#            if ti==0:
#                train_loss_au_b = np.expand_dims(temp_loss_au, axis=0)
#                train_pred_au_b = temp_pred_au
#            else:
#                train_loss_au_b = np.concatenate((train_loss_au_b, np.expand_dims(temp_loss_au, axis=0)), axis=0)
#                train_pred_au_b = np.concatenate((train_pred_au_b, temp_pred_au), axis=0)
        train_loss_au_b = 0
        np_f1_train_au =0
        
        
        perf_au_training.append([np.mean(train_loss_au_b), np_f1_train_au])
#        print('training')
#        print('AU total loss || ave f1')
#        print(np.mean(train_loss_au_b), np.mean(np_f1_train_au))
#        print(' ')
        
        #testing for each epo
        start_idx_test = np.arange(0, len(testD_path), 300)
        for ti in range(len(start_idx_test)):
            test_image, test_explabel, test_AUlabel, test_AUprob = get_train_batch(300, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
            
            _, temp_pred_au, temp_loss_au = right_AU_model.run(test_image, test_AUprob, load_AUconfig) #image, AUprob, AUconfig            
            
            if ti==0:
                test_loss_au_b = np.expand_dims(temp_loss_au, axis=0)
                test_pred_au_b = temp_pred_au
            else:
                test_loss_au_b = np.concatenate((test_loss_au_b, np.expand_dims(temp_loss_au, axis=0)), axis=0)
                test_pred_au_b = np.concatenate((test_pred_au_b, temp_pred_au), axis=0)
        
        np_f1_test_au = Compute_F1score_au(testL[:,1:], test_pred_au_b)
        perf_au_testing.append([np.mean(test_loss_au_b), np_f1_test_au])
        
#        print('testing')
#        print('AU total loss || ave f1')
#        print(np.mean(test_loss_au_b), np.mean(np_f1_test_au))
#        print(' ')
        
        if np.mean(np_f1_test_au) > best:
            best = np.mean(np_f1_test_au)
            print('best')
            print(best)
            file_name = title_string[:-2]
            write_model_path = os.path.join(model_folder, file_name)
            right_AU_model.save(define_model_path, write_model_path)
        
    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, title_string), "wb" )
    pickle.dump([perf_au_training, perf_au_testing], pickle_out)
    
    pickle_out.close()
    
#    file_name = title_string[:-2]
#    write_model_path = os.path.join(model_folder, file_name)
#    right_AU_model.save(define_model_path, write_model_path)
#    print('best')
#    print(best)
    right_AU_model.close()
    
#modify song
def read_label(path_index):
    labels=np.zeros([len(path_index),12],dtype = int)
    labels_configuration =np.zeros([len(path_index),2048],dtype = int)
    for i in range(len(path_index)):
        name=os.path.split(path_index[i])[-1]
        [file1, file2, file3] = name.split('_')
        number=int(file3.split('.')[0])
        data_label=sio.loadmat('/home/songtengfei/AU_OCC/'+file1+'_'+file2+'.mat')
        label_matrix=data_label['TT']
        diff = 1+number-label_matrix[1,0]
        labels[i,:] = label_matrix[1+diff,[1,2,4,6,7,10,12,14,15,17,23,24]]
        labels[i,1:12]=labels[i,[0,1,3,4,5,6,7,8,9,10,11]]
        if file2 == 'T1':
            labels[i,0]=0
        if file2 == 'T3':
            labels[i,0]=1
        if file2 == 'T5':
            labels[i,0]=2
        if file2 == 'T7':
            labels[i,0]=3
        if file2 == 'T8':
            labels[i,0]=4
        
        labels_configuration[i,:]=labels[i,1]*1+labels[i,2]*2+labels[i,3]*4+labels[i,4]*8+labels[i,5]*16+labels[i,6]*32+labels[i,7]*64+labels[i,8]*128+labels[i,9]*256+labels[i,10]*512+labels[i,11]*1024
    return labels, labels_configuration
fold3_path_label1, fold3_path_label1_config = read_label(fold1_index)
fold3_path_label2, fold3_path_label2_config = read_label(fold2_index) 
fold3_path_label3, fold3_path_label3_config = read_label(fold3_index)


name_string = 'initialAU-joint'
file1 = [name_string + '-fold1-new.p'][0]
file2 = [name_string + '-fold2-new.p'][0]
file3 = [name_string + '-fold3-new.p'][0]

main(fold1_index, fold2_index, fold3_index, fold1_label, fold2_label, fold3_label, file1)

main(fold1_index, fold3_index, fold2_index, fold1_label, fold3_label, fold2_label, file2)
main(fold2_index, fold3_index, fold1_index, fold2_label, fold3_label, fold1_label, file3)



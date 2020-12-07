import numpy as np
import pickle
import os
from helper_functions import get_train_batch
from ImportGraph import ImportRightAU
import scipy.io as sio
import pdb


os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  #(e.g. it will use GPU’s number 0 and 2.)
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
    path = ["../Dataset/MMI/%d-"%(session_name[i])+"%d"%(frame_name[i])+".jpg"]
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

def main(fold1,fold2,fold3,fold4,fold5,foldL1,foldL2,foldL3,foldL4,foldL5, title_string_au, title_string):
    trainD_path = np.concatenate([fold1,fold2,fold3,fold4],axis=0)
    testD_path = fold5
    trainL = np.concatenate([foldL1,foldL2,foldL3,foldL4],axis=0)
    testL = foldL5
    
    pickle_in = open(os.path.join( '../BNwithConstraints/PseudoBN/'+'list_AU_config.p'), "rb" )
    load_AUconfig = pickle.load(pickle_in)
    
    pickle_in = open( os.path.join( '../BNwithConstraints/PseudoBN/Total/'+ 'PGM_p.p'), "rb" )
    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    

    batch_size = 50
    epo = 100
    LR = 0.001
    best = -10
    
    perf_3rd_training = []
    perf_3rd_testing = []
    
    perf_K_training = []
    perf_K_testing = []
    
    total_samples = len(trainD_path)
    
    define_rightexp_file_name = 'define_right_exp'
    model_folder = os.path.join(os.path.join(os.getcwd()), 'Models')
    define_rightexp_path = os.path.join(model_folder, define_rightexp_file_name)
    
    AU_model_name = title_string_au
    AUmodel_path = os.path.join(model_folder, AU_model_name)
    right_AU_model = ImportRightAU(AUmodel_path)
    
    right_Exp_model = ImportRightExp(define_rightexp_path)
    for e in range(epo): #each epo go over all samples
    
        if e%20 == 0 and e!= 0:
            LR = LR*0.9
        order=np.random.permutation(len(trainD_path))
        trainD_path=trainD_path[order]
        trainL=trainL[order,:]
        start_idx = np.arange(0, total_samples, batch_size)
        for i in range(len(start_idx)): 
            
            index = start_idx[i]
            train_image, train_explabel, train_AUlabel, train_AUprob,_ = get_train_batch(batch_size, trainD_path, trainL, load_PGM_pAUconfig, index)
            
            np_p_au, _, _ = right_AU_model.run(train_image, train_AUprob, load_AUconfig) 
            
            right_Exp_model.train(np_p_au, train_explabel, LR, load_PGM_pExp)
            
        
        #testing for each epo
        start_idx_test = np.arange(0, len(testD_path), 300)
        for ti in range(len(start_idx_test)): 
            test_image, test_explabel, test_AUlabel, test_AUprob,_ = get_train_batch(300, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
            
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

        print('epo = %d, AU-based FER=%f, BN FER=%f'%(e, np_acc_test_exp_3rd, np_acc_test_exp_K))
        if np_acc_test_exp_3rd > best:
            best = np_acc_test_exp_3rd
            best_K = np_acc_test_exp_K
            write_model_path = os.path.join(model_folder, title_string)
            right_Exp_model.save(define_rightexp_path, write_model_path)
    
    
    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, title_string), "wb" )
    pickle.dump([perf_3rd_training, perf_3rd_testing, perf_K_training, perf_K_testing, best, best_K], pickle_out)
    
    pickle_out.close()
    
    file_name = title_string
    write_model_path = os.path.join(model_folder, file_name)
    right_Exp_model.save(define_rightexp_path, write_model_path)
    

method = 'pseudoTotal'
aufilename_srting1 = 'AU_model_5fold_1mmi_'+ method
aufilename_srting2 = 'AU_model_5fold_2mmi_'+ method 
aufilename_srting3 = 'AU_model_5fold_3mmi_'+ method 
aufilename_srting4 = 'AU_model_5fold_4mmi_'+ method 
aufilename_srting5 = 'AU_model_5fold_5mmi_'+ method 

expfilename_srting1 = 'RightExp_5fold_1mmi_'+ method
expfilename_srting2 = 'RightExp_5fold_2mmi_'+ method
expfilename_srting3 = 'RightExp_5fold_3mmi_'+ method
expfilename_srting4 = 'RightExp_5fold_4mmi_'+ method
expfilename_srting5 = 'RightExp_5fold_5mmi_'+ method


main(fold1_index,fold2_index,fold3_index,fold4_index,fold5_index,fold1_label,fold2_label,fold3_label,fold4_label,fold5_label, aufilename_srting1,  expfilename_srting1)
main(fold2_index,fold3_index,fold4_index,fold5_index,fold1_index,fold2_label,fold3_label,fold4_label,fold5_label,fold1_label, aufilename_srting2,  expfilename_srting2)
main(fold3_index,fold4_index,fold5_index,fold1_index,fold2_index,fold3_label,fold4_label,fold5_label,fold1_label,fold2_label, aufilename_srting3,  expfilename_srting3)
main(fold4_index,fold5_index,fold1_index,fold2_index,fold3_index,fold4_label,fold5_label,fold1_label,fold2_label,fold3_label, aufilename_srting4,  expfilename_srting4)
main(fold5_index,fold1_index,fold2_index,fold3_index,fold4_index,fold5_label,fold1_label,fold2_label,fold3_label,fold4_label, aufilename_srting5,  expfilename_srting5)


pickleFolder = os.path.join( os.getcwd(), 'Results')
pickle_in = open( os.path.join( pickleFolder, expfilename_srting1), "rb" )
_,_,_,_,best1,best_K1  = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, expfilename_srting2), "rb" )
_,_,_,_,best2,best_K2 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, expfilename_srting3), "rb" )
_,_,_,_,best3,best_K3 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, expfilename_srting4), "rb" )
_,_,_,_,best4,best_K4 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, expfilename_srting5), "rb" )
_,_,_,_,best5,best_K5 = pickle.load(pickle_in)


print('MMI:%s'%(method))
print('AU-based FER 1=%f, 2=%f, 3=%f, 4=%f,5=%f, ave=%f'%(best1, best2, best3, best4, best5, (best1+best2+best3+best4+best5)/5))
print('BN FER 1=%f, 2=%f, 3=%f, 4=%f,5=%f, ave=%f'%(best_K1, best_K2, best_K3, best_K4, best_K5, (best_K1+best_K2+best_K3+best_K4+best_K5)/5))


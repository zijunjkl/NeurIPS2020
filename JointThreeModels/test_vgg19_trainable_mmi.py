"""
Simple tester for the vgg19_trainable
"""

import numpy as np
import tensorflow as tf
import pickle
import os
from helper_functions_new import get_train_batch
from Expression_part_new import Loss_ExpressionModel_Labelonly
import pdb
import scipy.io as sio
import vgg19_trainable as vgg19
from sklearn.metrics import accuracy_score
import time

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #(e.g. it will use GPU’s number 0 and 2.)


def main(fold1,fold2,fold3,fold4,fold5,foldL1,foldL2,foldL3,foldL4,foldL5, title_string):
    print(title_string)
    pickleFolder = os.path.join( os.getcwd(), 'data_pickle') 
    
    trainD_path = np.concatenate([fold1,fold2,fold3,fold4],axis=0)
    testD_path = fold5
    trainL = np.concatenate([foldL1,foldL2,foldL3,foldL4],axis=0)
    testL = foldL5
    
    batch_size = 32
    epo = 20
    perf_exp_training=[]
    perf_exp_testing =[]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    images_norm = tf.map_fn(lambda frame:tf.image.per_image_standardization(frame),images)
    true_out = tf.placeholder(tf.int32, [None, ])
    train_mode = tf.placeholder(tf.bool)
    LR = tf.placeholder(tf.float32)
    # vgg = vgg19.Vgg19('./Models/vgg19.npy') # no pretraining
    vgg = vgg19.Vgg19('./Models/pre_trained-fer.npy') #pretrained on FER2013
    vgg.build(images_norm, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())
    
    # test classification
    # prob = sess.run(vgg.prob_exp, feed_dict={images: batch1, train_mode: False})
    # utils.print_prob(prob[0], './synset.txt')
    loss_exp = Loss_ExpressionModel_Labelonly(vgg.fc_exp, true_out)
    pred_Expression = tf.math.argmax(vgg.prob_exp, axis = 1)
    # with tf.name_scope("LOSS"):
    #     loss = loss_exp
    # tf.compat.v2.summary.scalar('loss',loss)
    # with tf.name_scope("ACC"):
    #     correct_prediction = tf.equal(pred_Expression, tf.argmax(true_out, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.compat.v2.summary.scalar('acc', accuracy)
    #merged_summary_op = tf.compat.v1.summary.merge_all()
    #summary_writer = tf.compat.v2.summary.SummaryWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())
    # simple 1-step training
    #cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    #train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    train = tf.train.GradientDescentOptimizer(LR).minimize(loss_exp)
#    train = tf.train.AdamOptimizer(LR).minimize(loss_exp)

    #train
    total_samples = len(trainD_path)
    trainL=trainL
    testL = testL
    max_acc=0    
    lr = 0.002
    for e in range(epo):
#        print(e)
        if e%20==0 and e!=0:
            lr = lr * 0.9
        order=np.random.permutation(len(trainD_path))
        train_data=np.array(trainD_path)[order]
        train_Label=trainL[order,:] 
        start_idx = np.arange(0, total_samples, batch_size)
        for i in range(len(start_idx)):  #batch
            index = start_idx[i]
            train_image, train_explabel = get_train_batch(batch_size, train_data, train_Label, index)
            time_one = time.time()
            sess.run(train, feed_dict={images: train_image, true_out: train_explabel, train_mode: True, LR: lr})
            time_diff= time.time()-time_one
            

        start_idx_train = np.arange(0, len(trainL), batch_size)
        for ti in range(len(start_idx_train)): 
            train_image, train_explabel = get_train_batch(batch_size, trainD_path, trainL, start_idx_train[ti])
            temp_loss, temp_pred = sess.run([loss_exp, pred_Expression], feed_dict={images: train_image, true_out: train_explabel, train_mode: False}) #p_exp, pred_exp, loss
            #summary_writer.add_summary(summary, e * len(start_idx) + i)
            if ti==0:
                train_loss_b = np.expand_dims(temp_loss, axis=0)
                train_pred_b = temp_pred
            else:
                train_loss_b = np.concatenate((train_loss_b, np.expand_dims(temp_loss, axis=0)), axis=0)
                train_pred_b = np.concatenate((train_pred_b, temp_pred), axis=0)
        np_train_loss_exp = np.mean(train_loss_b)
        score_train = accuracy_score(train_pred_b,trainL[:,0]-1)
        
        start_idx_test = np.arange(0, len(testL), batch_size)
        for ti in range(len(start_idx_test)): 
            test_image, test_explabel = get_train_batch(batch_size, testD_path, testL, start_idx_test[ti])
            temp_loss, temp_pred = sess.run([loss_exp, pred_Expression], feed_dict={images: test_image, true_out: test_explabel, train_mode: False}) #p_exp, pred_exp, loss
            
            if ti==0:
                test_loss_b = np.expand_dims(temp_loss, axis=0)
                test_pred_b = temp_pred
            else:
                test_loss_b = np.concatenate((test_loss_b, np.expand_dims(temp_loss, axis=0)), axis=0)
                test_pred_b = np.concatenate((test_pred_b, temp_pred), axis=0)
        np_test_loss_exp = np.mean(test_loss_b)
        score = accuracy_score(test_pred_b,testL[:,0]-1)
        
        print("epoch:%d,training:loss:%f,accuracy_training:%f,accuracy_testing:%f"%(e,np_train_loss_exp,score_train, score))
        if score > max_acc:
            max_acc = score
            print("testing max_score:%f",max_acc)
            vgg.save_npy(sess, './Models/'+title_string[:-2]+'.npy')
        
        np_train_loss_exp = np_train_loss_exp
        np_acc_train_exp =  (np.ravel(trainL[:,0]-1)== train_pred_b).sum()/len(trainL)
        perf_exp_training.append([np_train_loss_exp, np_acc_train_exp ])
        perf_exp_testing.append([np_test_loss_exp, score ])
                
    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, title_string), "wb" )
    pickle.dump([perf_exp_training, perf_exp_testing, max_acc], pickle_out)
    
    pickle_out.close()    
    
    

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

name_string = 'LeftExp'
file1 = [name_string + '-fold1_pre_mmi.p'][0]
file2 = [name_string + '-fold2_pre_mmi.p'][0]
file3 = [name_string + '-fold3_pre_mmi.p'][0]
file4 = [name_string + '-fold4_pre_mmi.p'][0]
file5 = [name_string + '-fold5_pre_mmi.p'][0]

main(fold1_index,fold2_index,fold3_index,fold4_index,fold5_index,fold1_label,fold2_label,fold3_label,fold4_label,fold5_label, file1)
main(fold2_index,fold3_index,fold4_index,fold5_index,fold1_index,fold2_label,fold3_label,fold4_label,fold5_label,fold1_label, file2)
main(fold3_index,fold4_index,fold5_index,fold1_index,fold2_index,fold3_label,fold4_label,fold5_label,fold1_label,fold2_label, file3)
main(fold4_index,fold5_index,fold1_index,fold2_index,fold3_index,fold4_label,fold5_label,fold1_label,fold2_label,fold3_label, file4)
main(fold5_index,fold1_index,fold2_index,fold3_index,fold4_index,fold5_label,fold1_label,fold2_label,fold3_label,fold4_label, file5)


pickleFolder = os.path.join( os.getcwd(), 'Results')
pickle_in = open( os.path.join( pickleFolder, file1), "rb" )
_,_,bestexp1  = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, file2), "rb" )
_,_,bestexp2 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, file3), "rb" )
_,_,bestexp3 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, file4), "rb" )
_,_,bestexp4 = pickle.load(pickle_in)
pickle_in = open( os.path.join( pickleFolder, file5), "rb" )
_,_,bestexp5 = pickle.load(pickle_in)


print('mmi')
print('FER 1=%f, 2=%f, 3=%f, 4=%f,5=%f, ave=%f'%(bestexp1,bestexp2, bestexp3, bestexp4, bestexp5, (bestexp1+bestexp2+bestexp3+bestexp4+bestexp5)/5))


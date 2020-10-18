"""
Simple tester for the vgg19_trainable
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
from helper_functions import Compute_F1score, Compute_F1score_au
from Expression_part_new import CNN_Expression, Loss_ExpressionModel, Loss_ExpressionModel_Labelonly
from Update_Posterior import Exp_KnowledgeModel
from ImportGraph import ImportRightAU, ImportRightExp, ImportLeftExp
#from Ipython.display import Image, display
import pdb
import itertools
import scipy.io as sio
import vgg19_trainable as vgg19
import utils
from sklearn.metrics import accuracy_score
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
img1 = utils.load_image("/home/zijun/Documents/VGG/test_data/tiger.jpeg")
img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

batch1 = img1.reshape((1, 224, 224, 3))

#def main(trainD_path, testD_path, train_label, testL, title_string):
def main(fold1_index, fold2_index, fold3_index, fold1_label, fold2_label, fold3_label,title_string):
    print(title_string)
    pickleFolder = os.path.join( os.getcwd(), 'data_pickle') 
    pickle_in = open( os.path.join( pickleFolder, 'list_AU_config3.p'), "rb" )
    load_AUconfig = pickle.load(pickle_in)
    pickle_in = open( os.path.join( pickleFolder, 'PGM_p-K2-MLE.p'), "rb" )
    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    LOGS_DIRECTORY="logs/train"

    trainD_path = list(itertools.chain(fold1_index, fold2_index))
    testD_path = fold3_index
    
    trainL = np.concatenate((fold1_label, fold2_label), axis=0)
    testL = fold3_label
    batch_size = 50
    epo = 100
    perf_exp_training=[]
    perf_exp_testing =[]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    images_norm = tf.map_fn(lambda frame:tf.image.per_image_standardization(frame),images)
    true_out = tf.placeholder(tf.int32, [None, ])
    train_mode = tf.placeholder(tf.bool)
    LR = tf.placeholder(tf.int32)
    #vgg = vgg19.Vgg19('./save_vgg/LeftExp-fold3.npy')
    #vgg = vgg19.Vgg19('./vgg19.npy')
    vgg = vgg19.Vgg19('./pre_trained-fer.npy') 
    vgg.build(images_norm, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())
    
    # test classification
    # prob = sess.run(vgg.prob_exp, feed_dict={images: batch1, train_mode: False})
    # utils.print_prob(prob[0], './synset.txt')
    loss_exp = Loss_ExpressionModel_Labelonly(vgg.prob_exp, true_out)
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

    #train
    total_samples = len(trainD_path)
    trainL=trainL
    testL = testL
    max_acc=0    
    lr = 0.012
    for e in range(epo):
        print(e)
        lr = lr * 0.8
        order=np.random.permutation(len(trainD_path))
        train_data=np.array(trainD_path)[order]
        train_Label=trainL[order,:] 
        start_idx = np.arange(0, total_samples, batch_size)
        for i in range(len(start_idx)): 
            index = start_idx[i]
            train_image, train_explabel = get_train_batch(batch_size, train_data, train_Label, load_PGM_pAUconfig, index)
            time_one = time.time()
            sess.run(train, feed_dict={images: train_image, true_out: train_explabel, train_mode: True, LR: lr})
            time_diff= time.time()-time_one
            

            if i%2==0 and i!=0:
                start_idx_train = np.arange(0, len(trainL), batch_size)
                for ti in range(len(start_idx_train)): 
                    train_image, train_explabel = get_train_batch(batch_size, trainD_path, trainL, load_PGM_pAUconfig, start_idx_train[ti])
                    temp_loss, temp_pred = sess.run([loss_exp, pred_Expression], feed_dict={images: train_image, true_out: train_explabel, train_mode: False}) #p_exp, pred_exp, loss
                    #summary_writer.add_summary(summary, e * len(start_idx) + i)
                    if ti==0:
                        train_loss_b = np.expand_dims(temp_loss, axis=0)
                        train_pred_b = temp_pred
                    else:
                        train_loss_b = np.concatenate((train_loss_b, np.expand_dims(temp_loss, axis=0)), axis=0)
                        train_pred_b = np.concatenate((train_pred_b, temp_pred), axis=0)
                np_train_loss_exp = np.mean(train_loss_b)
                score = accuracy_score(train_pred_b,trainL[:,0]-1)
#                print("epoch:%d,training:loss:%f,accuracy_training:%f, time:%f"%(e,np_train_loss_exp,score,time_diff))
            start_idx_test = np.arange(0, len(testL), batch_size)
            if i%2==0 and i!=0:
                for ti in range(len(start_idx_test)): 
                    test_image, test_explabel = get_train_batch(batch_size, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
                    temp_loss, temp_pred = sess.run([loss_exp, pred_Expression], feed_dict={images: test_image, true_out: test_explabel, train_mode: False}) #p_exp, pred_exp, loss
                    
                    if ti==0:
                        test_loss_b = np.expand_dims(temp_loss, axis=0)
                        test_pred_b = temp_pred
                    else:
                        test_loss_b = np.concatenate((test_loss_b, np.expand_dims(temp_loss, axis=0)), axis=0)
                        test_pred_b = np.concatenate((test_pred_b, temp_pred), axis=0)
                np_test_loss_exp = np.mean(test_loss_b)
                score = accuracy_score(test_pred_b,testL[:,0]-1)
                
#                print("epoch:%d,testing:loss:%f,accuracy_testing:%f"%(e,np_test_loss_exp,score))
#                print(i)
                #pdb.set_trace()
                if score > max_acc:
                    max_acc = score
                    print("testing max_score:%f",max_acc)
                    if score > 0.50:
                       vgg.save_npy(sess, './save_vgg/'+title_string[:-2]+'.npy')
                
                np_train_loss_exp = np_train_loss_exp
                np_acc_train_exp =  (np.ravel(trainL[:,0]-1)== train_pred_b).sum()/len(trainL)
                #np_f1_train_exp = Compute_F1score(trainL[:,0]-1, train_pred_b)
                perf_exp_training.append([np_train_loss_exp, np_acc_train_exp ])
#                print("epoch:%d,training ACC:%f"%(e,np_acc_train_exp ))
                perf_exp_testing.append([np_test_loss_exp, score ])
                
    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, title_string), "wb" )
    pickle.dump([perf_exp_training, perf_exp_testing], pickle_out)
    
    pickle_out.close()
    
    # test save
    #vgg.save_npy(sess, './test-save.npy')
    
    


pickleFolder = os.path.join( os.getcwd(), 'data_pickle') 
#pickle_in = open( os.path.join( pickleFolder, '3fold-image-serverpath.p'), "rb" )
#trainD_path, validD_path, testD_path = pickle.load(pickle_in)
#train1_index = trainD_path+validD_path
#pickle_in = open(os.path.join(pickleFolder, '3fold-exp-5AU-label.p'), "rb")
#trainL, validL, testL = pickle.load(pickle_in)
#train_label=np.concatenate((trainL, validL), axis=0)

#BP4D
data = sio.loadmat(os.path.join( pickleFolder,'BP4D_Apex_11AU.mat'))
index = data['BP4D_Apex_11AU']
sub_name = index[0,0]['SUB']
task_name = index[0,0]['TASK']
image_name = index[0,0]['IMGIND']
label_exp = index[0,0]['EXP']
label_AU = index[0,0]['AU']
data = sio.loadmat(os.path.join( pickleFolder,'BP4D_Apex_732.mat'))
index = data['BP4D_Apex_732']
sub_name = index[0,0]['SUB']
task_name = index[0,0]['TASK']
image_name = index[0,0]['IMGIND']
#label_exp = index[0,0]['EXP']
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

#label_ = sio.loadmat('BP4D_8AU.mat')
#label_AU = label_['AU_8']
fold1_index=path_all[0:248]
fold2_index=path_all[248:492]
fold3_index=path_all[492:732]
fold1_label=np.concatenate((label_exp[0:248], label_AU[0:248, [0,1,5,6,9]] - 1), axis = 1)
fold2_label=np.concatenate((label_exp[248:492], label_AU[248:492, [0,1,5,6,9]] -1), axis = 1)
fold3_label=np.concatenate((label_exp[492:732], label_AU[492:732, [0,1,5,6,9]] -1), axis = 1)

name_string = 'LeftExp'
file1 = [name_string + '-fold1_pre_bp4d_3rd.p'][0]
file2 = [name_string + '-fold2_pre_bp4d_3rd.p'][0]
file3 = [name_string + '-fold3_pre_bp4d_3rd.p'][0]
#main(path_all[0:492], fold3_index, np.concatenate((fold1_label,fold2_label),axis=0), fold3_label, file1)
main(fold1_index, fold2_index, fold3_index, fold1_label, fold2_label, fold3_label, file1)
main(fold1_index, fold3_index, fold2_index, fold1_label, fold3_label, fold2_label, file2)
main(fold2_index, fold3_index, fold1_index, fold2_label, fold3_label, fold1_label, file3)

#ck+
#pdb.set_trace()


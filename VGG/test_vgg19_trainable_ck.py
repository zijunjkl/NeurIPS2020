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
from helper_functions import Compute_F1score, Compute_F1score_au,get_train_batch_ck224
from Expression_part_new import CNN_Expression, Loss_ExpressionModel, Loss_ExpressionModel_Labelonly
from Update_Posterior import Exp_KnowledgeModel
from ImportGraph import ImportRightAU, ImportRightExp, ImportLeftExp
#from Ipython.display import Image, display
import pdb
import itertools
import scipy.io as sio
import vgg19_trainable_ck as vgg19
import utils
from sklearn.metrics import accuracy_score
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
img1 = utils.load_image("/home/zijun/Documents/VGG/test_data/tiger.jpeg")
img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

batch1 = img1.reshape((1, 224, 224, 3))

#def main(trainD_path, testD_path, train_label, testL, title_string):
def main(fold1_index, fold2_index, fold3_index, fold4_index, fold5_index, fold1_label, fold2_label, fold3_label, fold4_label, fold5_label, title_string):
    print(title_string)
    pickleFolder = os.path.join( os.getcwd(), 'data_pickle') 
    pickle_in = open( os.path.join( '/home/zijun/Documents/CK+/PGMmodels/BP4D-to-CK-8AU/6exp-8au', 'list_AU_config.p'), "rb" )
    load_AUconfig = pickle.load(pickle_in)
    pickle_in = open( os.path.join( '/home/zijun/Documents/CK+/PGMmodels/BP4D-to-CK-8AU/6exp-8au', 'PGM_p-K2-Bayes.p'), "rb" )
    load_PGM_pExp, load_PGM_pAUconfig = pickle.load(pickle_in)
    LOGS_DIRECTORY="logs/train"

    trainD_path = list(itertools.chain(fold1_index, fold2_index, fold3_index, fold4_index))
    testD_path = fold5_index
    
    trainL = np.concatenate((fold1_label, fold2_label, fold3_label, fold4_label), axis=0)
    testL = fold5_label
    batch_size = 64
    epo = 100
    learn_rate =0.01
    perf_exp_training=[]
    perf_exp_testing =[]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    images_norm = tf.map_fn(lambda frame:tf.image.per_image_standardization(frame),images)
    true_out = tf.placeholder(tf.int32, [None, ])
    train_mode = tf.placeholder(tf.bool)
    #vgg = vgg19.Vgg19('./save_vgg/LeftExp-fold2_ck.npy')
    #vgg = vgg19.Vgg19('./save_vgg/LeftExp-fold5_pre_ck.npy')
    vgg = vgg19.Vgg19('./vgg19.npy')
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
    train = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss_exp)

    #train
    total_samples = len(trainD_path)
    max_acc=0    
    for e in range(epo):
        if e%10==0 and e!=0:
            learn_rate=learn_rate*0.9
        order=np.random.permutation(len(trainD_path))
        train_data=np.array(trainD_path)[order]
        train_Label=trainL[order,:] 
        start_idx = np.arange(0, total_samples, batch_size)
        for i in range(len(start_idx)): 
            index = start_idx[i]
            train_image, train_explabel,_,_ = get_train_batch_ck224(batch_size, train_data, train_Label, load_PGM_pAUconfig, index)
            time_one = time.time()
            sess.run(train, feed_dict={images: train_image, true_out: train_explabel, train_mode: True})
            time_diff= time.time()-time_one
            

            if i%2==0 and i!=0:
                start_idx_train = np.arange(0, len(trainL), batch_size)
                for ti in range(len(start_idx_train)): 
                    train_image, train_explabel,_,_ = get_train_batch_ck224(batch_size, trainD_path, trainL, load_PGM_pAUconfig, start_idx_train[ti])
                    temp_loss, temp_pred = sess.run([loss_exp, pred_Expression], feed_dict={images: train_image, true_out: train_explabel, train_mode: False}) #p_exp, pred_exp, loss
                    #summary_writer.add_summary(summary, e * len(start_idx) + i)
                    if ti==0:
                        train_loss_b = np.expand_dims(temp_loss, axis=0)
                        train_pred_b = temp_pred
                    else:
                        train_loss_b = np.concatenate((train_loss_b, np.expand_dims(temp_loss, axis=0)), axis=0)
                        train_pred_b = np.concatenate((train_pred_b, temp_pred), axis=0)
                np_train_loss_exp = np.mean(train_loss_b)
                score = accuracy_score(train_pred_b,trainL[:,0]-2)
                print("epoch:%d,training:loss:%f,accuracy_training:%f, time:%f"%(e,np_train_loss_exp,score,time_diff))
                
            start_idx_test = np.arange(0, len(testL), batch_size)
            if i%2==0 and i!=0:
                for ti in range(len(start_idx_test)): 
                    test_image, test_explabel,_,_ = get_train_batch_ck224(batch_size, testD_path, testL, load_PGM_pAUconfig, start_idx_test[ti])
                    temp_loss, temp_pred = sess.run([loss_exp, pred_Expression], feed_dict={images: test_image, true_out: test_explabel, train_mode: False}) #p_exp, pred_exp, loss
                    
                    if ti==0:
                        test_loss_b = np.expand_dims(temp_loss, axis=0)
                        test_pred_b = temp_pred
                    else:
                        test_loss_b = np.concatenate((test_loss_b, np.expand_dims(temp_loss, axis=0)), axis=0)
                        test_pred_b = np.concatenate((test_pred_b, temp_pred), axis=0)
                np_test_loss_exp = np.mean(test_loss_b)
                score = accuracy_score(test_pred_b,testL[:,0]-2)
                
                print("epoch:%d,testing:loss:%f,accuracy_testing:%f"%(e,np_test_loss_exp,score))
                #pdb.set_trace()
                print(i)
                if max_acc < score:
                    max_acc = score
                    print("max_score:%f",max_acc)
                    if score > 0.8:
                        vgg.save_npy(sess, './save_vgg/'+title_string[:-2]+'.npy')
                
                np_train_loss_exp = np_train_loss_exp
                np_acc_train_exp =  (np.ravel(trainL[:,0]-2)== train_pred_b).sum()/len(trainL)
                #np_f1_train_exp = Compute_F1score(trainL[:,0]-1, train_pred_b)
                perf_exp_training.append([np_train_loss_exp, np_acc_train_exp ])
                print("epoch:%d,training ACC:%f"%(e,np_acc_train_exp ))
                perf_exp_testing.append([np_test_loss_exp, score ])
                
    pickleFolder = os.path.join(os.getcwd())
    result_folder = os.path.join(pickleFolder, 'Results')
    
    pickle_out = open( os.path.join( result_folder, title_string), "wb" )
    pickle.dump([perf_exp_training, perf_exp_testing], pickle_out)
    
    pickle_out.close()
    
    # test save
    #vgg.save_npy(sess, './test-save.npy')
    
    


pickleFolder = os.path.join( os.getcwd(), 'data_pickle') 
pickle_in = open( os.path.join( pickleFolder, '3fold-image-serverpath.p'), "rb" )
trainD_path, validD_path, testD_path = pickle.load(pickle_in)
train1_index = trainD_path+validD_path
pickle_in = open(os.path.join(pickleFolder, '3fold-exp-5AU-label.p'), "rb")
trainL, validL, testL = pickle.load(pickle_in)
train_label=np.concatenate((trainL, validL), axis=0)

#CK+
data=sio.loadmat('/home/zijun/Documents/CK+/'+'CK+_6_BASIC.mat')
path=data['path']
label=data['labels']
for i in range(label.shape[0]):
    if label[i,0]==1:
        label[i,0]=label[i,0]+1
fold1_in=path[0:64]
fold2_in=path[64:137]
fold3_in=path[137:193]
fold4_in=path[193:244]
fold5_in=path[244:309]
fold1_la=label[0:64,:]
fold2_la=label[64:137,:]
fold3_la=label[137:193,:]
fold4_la=label[193:244,:]
fold5_la=label[244:309,:]

name_string = 'LeftExp'
file1 = [name_string + '-fold1_npre_ck.p'][0]
file2 = [name_string + '-fold2_npre_ck.p'][0]
file3 = [name_string + '-fold3_npre_ck.p'][0]
file4 = [name_string + '-fold4_npre_ck.p'][0]
file5 = [name_string + '-fold5_npre_ck.p'][0]

#main(fold1_in, fold2_in, fold3_in, fold4_in, fold5_in, fold1_la, fold2_la, fold3_la, fold4_la, fold5_la, file1)
#main(fold5_in, fold1_in, fold2_in, fold3_in, fold4_in, fold5_la, fold1_la, fold2_la, fold3_la, fold4_la, file2)
main(fold4_in, fold5_in, fold1_in, fold2_in, fold3_in, fold4_la, fold5_la, fold1_la, fold2_la, fold3_la, file3)
main(fold3_in, fold4_in, fold5_in, fold1_in, fold2_in, fold3_la, fold4_la, fold5_la, fold1_la, fold2_la, file4)
main(fold2_in, fold3_in, fold4_in, fold5_in, fold1_in, fold2_la, fold3_la, fold4_la, fold5_la, fold1_la, file5)

#ck+

pdb.set_trace()


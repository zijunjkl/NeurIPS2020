#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:29:07 2019

@author: zijun
"""

import pickle
import numpy as np
import os
import matplotlib.image as mping
import matplotlib.pyplot as plt
pickleFolder = os.path.join( os.getcwd()) 

pickle_in = open( os.path.join( pickleFolder, 'LeftExp-fold3_pre_bp4d_2nd.npy'), "rb" )
#perf_train_au1, perf_test_au1 = pickle.load(pickle_in)
best = pickle.load(pickle_in)
##from Ipython.display import Image, display
#pickleFolder = os.path.join( os.getcwd()) 
#pickle_in = open( os.path.join( pickleFolder, 'Model-AU-image-3layerCNN-10232019_fold_1.p'), "rb" )
##pickle_in = open( os.path.join( pickleFolder, 'Expression-3layerCNN-1022-peak-fold1.p'), "rb" )
#perf_train_exp1, perf_test_exp1 = pickle.load(pickle_in)
#
#pickleFolder = os.path.join( os.getcwd()) 
#pickle_in = open( os.path.join( pickleFolder, 'Model-AU-image-3layerCNN-10232019_fold_2.p'), "rb" )
##pickle_in = open( os.path.join( pickleFolder, 'Expression-3layerCNN-1022-peak-fold2.p'), "rb" )
#perf_train_exp2, perf_test_exp2 = pickle.load(pickle_in)
#
#pickleFolder = os.path.join( os.getcwd()) 
#pickle_in = open( os.path.join( pickleFolder, 'Model-AU-image-3layerCNN-10232019_fold_3.p'), "rb" )
##pickle_in = open( os.path.join( pickleFolder, 'Expression-3layerCNN-1022-peak-fold3.p'), "rb" )
#perf_train_exp3, perf_test_exp3 = pickle.load(pickle_in)
#
#perf_train = [perf_train_exp1[-1], perf_train_exp2[-1], perf_train_exp3[-1]]
#f1_train = np.mean(np.asarray(perf_train)[:,[1,2,3,4,5]], axis=0)
#
#perf = [perf_test_exp1[-1], perf_test_exp2[-1], perf_test_exp3[-1]]
#f1 = np.mean(np.asarray(perf)[:,[1,2,3,4,5]], axis=0)

pickleFolder = os.path.join( os.getcwd()) 
pickle_in = open( os.path.join( pickleFolder, 'LeftExp-fold1_pre_ck.npy'), "rb" )
#perf_train_au1, perf_test_au1 = pickle.load(pickle_in)
perf_train_exp1, perf_test_exp1, perf_train_au1, perf_test_au1, perf_posterior1 = pickle.load(pickle_in)

pickle_in = open( os.path.join( pickleFolder, 'LeftExp-fold2_pre_ck.npy'), "rb" )
#perf_train_au1, perf_test_au1 = pickle.load(pickle_in)
perf_train_exp1, perf_test_exp2, perf_train_au1, perf_test_au1, perf_posterior1 = pickle.load(pickle_in)

pickle_in = open( os.path.join( pickleFolder, 'LeftExp-fold3_pre_ck.npy'), "rb" )
#perf_train_au1, perf_test_au1 = pickle.load(pickle_in)
perf_train_exp1, perf_test_exp3, perf_train_au1, perf_test_au1, perf_posterior1 = pickle.load(pickle_in)

pickle_in = open( os.path.join( pickleFolder, 'LeftExp-fold4_pre_ck.npy'), "rb" )
#perf_train_au1, perf_test_au1 = pickle.load(pickle_in)
perf_train_exp1, perf_test_exp4, perf_train_au1, perf_test_au1, perf_posterior1 = pickle.load(pickle_in)

pickle_in = open( os.path.join( pickleFolder, 'LeftExp-fold5_pre_ck.npy'), "rb" )
#perf_train_au1, perf_test_au1 = pickle.load(pickle_in)
perf_train_exp1, perf_test_exp5, perf_train_au1, perf_test_au1, perf_posterior1 = pickle.load(pickle_in)

test_exp1 = np.asarray(perf_test_exp1)
test_exp2 = np.asarray(perf_test_exp2)
test_exp3 = np.asarray(perf_test_exp3)
test_exp4 = np.asarray(perf_test_exp4)
test_exp5 = np.asarray(perf_test_exp5)

ave = np.max(test_exp1[:,0]) + np.max(test_exp2[:,0]) + np.max(test_exp3[:,0]) + np.max(test_exp4[:,0]) + np.max(test_exp5[:,0])
print(ave/5)
#pickleFolder = os.path.join( os.getcwd()) 
#pickle_in = open( os.path.join( pickleFolder, 'fold2.p'), "rb" )
##perf_train_au2, perf_test_au2 = pickle.load(pickle_in)
#perf_train_exp2, perf_test_exp2, perf_train_au2, perf_test_au2, perf_posterior2 = pickle.load(pickle_in)
#
#pickleFolder = os.path.join( os.getcwd()) 
#pickle_in = open( os.path.join( pickleFolder, 'fold3.p'), "rb" )
##perf_train_au3, perf_test_au3 = pickle.load(pickle_in)
#perf_train_exp3, perf_test_exp3, perf_train_au3, perf_test_au3, perf_posterior3 = pickle.load(pickle_in)

##plot loss on training and validation
#loss_testing_exp = np.asarray(perf_posterior1)
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_testing_exp[:,0])), loss_testing_exp[:,0],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,1])), loss_testing_exp[:,1],'g')
##plt.plot(np.arange(0, len(loss_testing_exp[:,6])), loss_testing_exp[:,6],'y')
#plt.title('Acc for each epo for Expression: Testing')
#plt.xlabel('epo')
#plt.ylabel('Acc')
#plt.ylim(0,1)

#plot loss on training and validation
#loss_testing_exp = np.asarray(perf_test_exp2)
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_testing_exp[:,0])), loss_testing_exp[:,0],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,2])), loss_testing_exp[:,2],'g')
#plt.plot(np.arange(0, len(loss_testing_exp[:,4])), loss_testing_exp[:,4],'b')
##plt.plot(np.arange(0, len(loss_testing_exp[:,6])), loss_testing_exp[:,6],'y')
#plt.title('Acc for each epo for Expression: Testing')
#plt.xlabel('epo')
#plt.ylabel('Acc')
#plt.ylim(0,1)
#plt.legend(['red: Left exp','green:Right exp', 'blue: PGM', 'yellow: posterior'], loc = 'upper right')
#
##plot loss on training and validation
#loss_testing_exp = np.reshape(np.asarray(perf_test_au2), [1600,5])
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_testing_exp[:,0])), loss_testing_exp[:,0],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,1])), loss_testing_exp[:,1],'g')
#plt.plot(np.arange(0, len(loss_testing_exp[:,2])), loss_testing_exp[:,2],'b')
#plt.plot(np.arange(0, len(loss_testing_exp[:,3])), loss_testing_exp[:,3],'y')
#plt.plot(np.arange(0, len(loss_testing_exp[:,4])), loss_testing_exp[:,4],'k')
#plt.title('F1 for each epo for each AU: Testing')
#plt.xlabel('epo')
#plt.ylabel('F1')
#plt.ylim(0,1)
#plt.legend(['red: AU1','green:AU2', 'blue: AU6', 'yellow: AU7', 'black: AU10'], loc = 'lower right')
#
#perf = [perf_test_exp1[-1], perf_test_exp2[-1], perf_test_exp3[-1]]
#
#accuracy = np.mean(np.asarray(perf)[:,0])
#print('left expression %f' %accuracy)
#
#accuracy = np.mean(np.asarray(perf)[:,2])
#print('right expression %f' %accuracy)
#
#accuracy = np.mean(np.asarray(perf)[:,4])
#print('PGM expression %f' %accuracy)
#
#perf = [perf_posterior1[-1], perf_posterior2[-1], perf_posterior3[-1]]
#accuracy = np.mean(np.asarray(perf)[:,1])
#print('posterior expression %f' %accuracy)
#
#aa = perf_test_au1[-1]
#bb = perf_test_au2[-1]
#cc = perf_test_au3[-1]
#
#f1_au = np.mean(np.asarray([aa,bb,cc]), axis=0)
#print(f1_au)
#
##loss_train = np.zeros([len(perf_train_au2),1])
##loss_test = np.zeros([len(perf_test_au2),1])
#
###fold 1 best
#f1_train = np.zeros([len(perf_train_au1),5])
#f1_train_ave = np.zeros([len(perf_train_au1),1])
#f1_test = np.zeros([len(perf_test_au1),5])
#f1_test_ave = np.zeros([len(perf_test_au1),1])
#for i in range(len(perf_train_au1)):
#    f1_train[i,:] = perf_train_au1[i]#[1]
#    f1_train_ave[i] = np.mean(f1_train[i,:])
#    f1_test[i,:] = perf_test_au1[i]#[1]
#    f1_test_ave[i] = np.mean(f1_test[i,:])
#
#idx1 = 20+np.argmax(f1_test_ave[20:-1])
#fold1_train = f1_train[idx1,:]
#fold1_test = f1_test[idx1,:]
#
####fold 2 best
#f1_train = np.zeros([len(perf_train_au2),5])
#f1_train_ave = np.zeros([len(perf_train_au2),1])
#f1_test = np.zeros([len(perf_test_au2),5])
#f1_test_ave = np.zeros([len(perf_test_au2),1])
#for i in range(len(perf_train_au2)):
#    f1_train[i,:] = perf_train_au2[i]#[1]
#    f1_train_ave[i] = np.mean(f1_train[i,:])
#    f1_test[i,:] = perf_test_au2[i]#[1]
#    f1_test_ave[i] = np.mean(f1_test[i,:])
#
#idx2 = 20+np.argmax(f1_test_ave[20:-1])
#fold2_train = f1_train[idx2,:]
#fold2_test = f1_test[idx2,:]
#
####fold 3 best
#f1_train = np.zeros([len(perf_train_au3),5])
#f1_train_ave = np.zeros([len(perf_train_au3),1])
#f1_test = np.zeros([len(perf_test_au3),5])
#f1_test_ave = np.zeros([len(perf_test_au3),1])
#for i in range(len(perf_train_au3)):
#    f1_train[i,:] = perf_train_au3[i]#[1]
#    f1_train_ave[i] = np.mean(f1_train[i,:])
#    f1_test[i,:] = perf_test_au3[i]#[1]
#    f1_test_ave[i] = np.mean(f1_test[i,:])
#
#idx3 = 20+np.argmax(f1_test_ave[20:-1])
#fold3_train = f1_train[idx3,:]
#fold3_test = f1_test[idx3,:]
##
#f1_train = np.mean(np.asarray([fold1_train,fold2_train,fold3_train]), axis=0)
#f1_test = np.mean(np.asarray([fold1_test, fold2_test, fold3_test]), axis=0)
#print(f1_train)
#print(f1_test)
##plot loss on training and validation
#loss_training_exp = loss_train
#loss_testing_exp = loss_test
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_training_exp)), loss_training_exp,'r')
#plt.plot(np.arange(0, len(loss_testing_exp)), loss_testing_exp,'g')
#plt.title('Loss for each epo for expression')
#plt.xlabel('epo')
#plt.ylabel('Loss')
#plt.legend(['red: training data','green:testing data'], loc = 'upper right')
#
##plot loss on training and validation
#loss_training_exp = np.asarray(perf_train_exp)
#loss_testing_exp = np.asarray(perf_test_exp)
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_training_exp[:,1])), loss_training_exp[:,1],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,1])), loss_testing_exp[:,1],'g')
#plt.title('Accuracy for each epo for expression')
#plt.xlabel('epo')
#plt.ylabel('Acc')
#plt.legend(['red: training data','green:testing data'], loc = 'upper right')

##plot loss on training and validation
#loss_testing_exp = f1_train
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_testing_exp[:,0])), loss_testing_exp[:,0],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,1])), loss_testing_exp[:,1],'g')
#plt.plot(np.arange(0, len(loss_testing_exp[:,2])), loss_testing_exp[:,2],'b')
#plt.plot(np.arange(0, len(loss_testing_exp[:,3])), loss_testing_exp[:,3],'y')
#plt.plot(np.arange(0, len(loss_testing_exp[:,4])), loss_testing_exp[:,4],'k')
#plt.title('F1 for each epo for 5 AUs: Training')
#plt.xlabel('epo')
#plt.ylabel('f1')
#plt.ylim(0,1)
#plt.legend(['red: AU1','green:AU2', 'blue: AU6', 'yellow: AU7', 'black: AU10'], loc = 'upper right')
#
#loss_testing_exp = f1_test
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_testing_exp[:,0])), loss_testing_exp[:,0],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,1])), loss_testing_exp[:,1],'g')
#plt.plot(np.arange(0, len(loss_testing_exp[:,2])), loss_testing_exp[:,2],'b')
#plt.plot(np.arange(0, len(loss_testing_exp[:,3])), loss_testing_exp[:,3],'y')
#plt.plot(np.arange(0, len(loss_testing_exp[:,4])), loss_testing_exp[:,4],'k')
#plt.title('F1 for each epo for 5 AUs: Testing')
#plt.xlabel('epo')
#plt.ylabel('f1')
#plt.ylim(0,1)
#plt.legend(['red: AU1','green:AU2', 'blue: AU6', 'yellow: AU7', 'black: AU10'], loc = 'upper right')

#loss_testing_exp = np.asarray(perf_test_exp3)
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_testing_exp[:,1])), loss_testing_exp[:,1],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,2])), loss_testing_exp[:,2],'g')
#plt.plot(np.arange(0, len(loss_testing_exp[:,3])), loss_testing_exp[:,3],'b')
#plt.plot(np.arange(0, len(loss_testing_exp[:,4])), loss_testing_exp[:,4],'y')
#plt.plot(np.arange(0, len(loss_testing_exp[:,5])), loss_testing_exp[:,5],'k')
#plt.title('F1 for each epo for 5 AUs')
#plt.xlabel('epo')
#plt.ylabel('Acc')
#plt.legend(['red: AU1','green:AU2', 'blue: AU6', 'yellow: AU7', 'black: AU10'], loc = 'upper right')


##plot loss on training and validation
#loss_training_exp = np.asarray(perf_train_exp)
#loss_testing_exp = np.asarray(perf_test_exp)
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_training_exp[:,0])), loss_training_exp[:,0],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,0])), loss_testing_exp[:,0],'g')
#plt.title('Loss for each epoch for expression')
#plt.xlabel('epo')
#plt.ylabel('Loss')
#plt.legend(['red: training data','green:testing data'], loc = 'upper right')

##plot loss on training and validation
#loss_training_au = np.asarray(perf_train_au)[:,0]
#loss_testing_au = np.asarray(perf_test_au)[:,0]
#loss_training_au = np.concatenate(loss_training_au).astype(None)
#fig = plt.figure(figsize = (12,5))
#plt.plot(np.arange(0, len(loss_training_exp[:,0])), loss_training_exp[:,0],'r')
#plt.plot(np.arange(0, len(loss_testing_exp[:,0])), loss_testing_exp[:,0],'g')
#plt.title('Loss for each epoch')
#plt.xlabel('epo')
#plt.ylabel('Loss')
#plt.legend(['red: training data','greed:testing data'], loc = 'upper right')

#train_f1 = perf_train[-1][1]
#validation_f1 = perf_validation[-1][1]
#test_f1 = perf_test[-1][1]
#print(train_f1)
#print(validation_f1)
#print(test_f1)
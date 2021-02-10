#-*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:48:22 2018

@author: Aite Zhao
"""

from __future__ import print_function
#import random
import tensorflow as tf
#from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
#import plot_confusion_matrix
import rnn_cell_GRU as rnn_cell
import rnn
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
#from EvoloPy import *
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from hmmlearn import hmm
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.learning_curve import validation_curve
from sklearn.svm import SVC
from hmmexm import hmm_4model_classification,hmm_3model_classification
from sklearn.model_selection import LeaveOneOut, KFold,cross_val_score
from deep_CCA_model import *
from linear_cca import linear_cca

n_classes = 52

def labelprocess(label,n_class=n_classes):
    label_length=len(label)
    label_matrix=np.zeros((label_length,n_class))
    for i,j in enumerate(label): 
       label_matrix[i,int(j)]=1
    return label_matrix
    
#    
#def kfold_validation(data,label,n_splits=5):
#    #   K fold cross validation
#    x_trains = []
#    y_trains = []
#    x_tests = []
#    y_tests = []
#    k_fold = KFold(n_splits)
#    for train_index, test_index in k_fold.split(data):
#        X_train, X_test = data[train_index], data[test_index]
#        y_train, y_test = label[train_index], label[test_index]
#        x_trains.append(X_train)
#        y_trains.append(y_train)
#        x_tests.append(X_test)
#        y_tests.append(y_test)
#    return x_trains,y_trains,x_tests,y_tests

#

def next_batch(batch_size,train_x,train_y,newli_train,force):
    global batchid_force, batchid_time
    if force==True:
        if batchid_force+batch_size > len(train_x):
           batchid_force = 0
        batch_data = (train_x[batchid_force:min(batchid_force +batch_size, len(train_y)),:])
        batch_labels = (newli_train[batchid_force:min(batchid_force + batch_size, len(newli_train)),:])
        batch_labels_1d = (train_y[batchid_force:min(batchid_force + batch_size, len(train_y))])
        batchid_force = min(batchid_force + batch_size, len(train_y))
        return batch_data, batch_labels,batch_labels_1d
    else:
        if batchid_time+batch_size > len(train_x):
           batchid_time = 0
        batch_data = (train_x[batchid_time:min(batchid_time +batch_size, len(train_y)),:])
        batch_labels = (newli_train[batchid_time:min(batchid_time + batch_size, len(newli_train)),:])
        batch_labels_1d = (train_y[batchid_time:min(batchid_time + batch_size, len(train_y))])
        batchid_time = min(batchid_time + batch_size, len(train_y))
        return batch_data, batch_labels,batch_labels_1d



def RNN(x, weights, biases, n_input):
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(tensor=x, shape=[-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(value=x, num_or_size_splits=n_steps, axis=0)
    # Define a lstm cell with tensorflow
    #lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1)
    lstm_cell = rnn_cell.GRUCell(n_hidden)
    #lstm_cell = rnn_cell.LSTMCell(n_hidden,use_peepholes=True)
    # avoid overfitting
    lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    # 2 layers lstm
#    num_units = [256, 256]
#    cells = [rnn_cell.GRUCell(num_units=n) for n in num_units]
#    lstm_cell = rnn_cell.MultiRNNCell(cells)
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * 2)   
    # Get lstm cell output
#    print(x)
    outputs, states = rnn.rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases, outputs


def feature_connect(a_time,a_force):
    a=np.array([])
    for j in range(int(11340/15)):
        f=np.array([])
        for i in range(15):
            f = np.concatenate((f, a_force[j*15+i,:]), axis=0) if f.size else a_force[j*15+i,:]
        a=np.c_[a,f] if a.size else f    
#    np.savetxt('./feature_extract/fusionfeature_data.txt', np.c_[a_time,np.transpose(a)],fmt='%.4f')
    return np.c_[a_time,np.transpose(a)],np.transpose(a)

def DCCA():
    #   LSTM CCA
    outdim_size = 10
    input_size1 = n_hidden
    input_size2 = n_hidden
#    input_size2 = 256
    layer_sizes1 = [1024, 1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, 1024, outdim_size]
    
    layer_sizes3 = [1024, 1024, 1024, n_classes]
    layer_sizes4 = [1024, 1024, 1024, n_classes]
    reg_par = 1e-4
    use_all_singular_values = True
    dcca_model = DeepCCA(layer_sizes1, layer_sizes2,layer_sizes3,layer_sizes4,
                          input_size1, input_size2,
                          outdim_size,
                          reg_par, use_all_singular_values)
    
    return dcca_model

def softmax(x): 
    x_exp = np.exp(x) 
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) 
    s = x_exp / x_sum 
    return s


if __name__=='__main__':
    #remove cpu occupation
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""

    
#   load data 
    a_force=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/sdugait/12023f.txt")
    a_force=a_force[:,0:60]
    b_force=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/sdugait/12023label.txt")
    b_force=b_force-1
    
    a_time=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/results/sdu/feature/out256_sdu_img.txt")
    b_time=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/sdugait/12023label.txt")
#    a_time=a_time[:,270:330]
    b_time=b_time-1
#    a_time=preprocessing.normalize(a_time+1) 
    
    
    
    all_fea_force=labelprocess(b_force)
    all_fea_time=labelprocess(b_time)
    
    
##   train_test_split 20% testing
#    train_x_time,test_x_time,train_y_time,test_y_time = train_test_split(a_time,b_time,test_size=0.2)
#    train_x_force,test_x_force,train_y_force,test_y_force = train_test_split(a_force,b_force,test_size=0.2)
#    print(train_x_time.shape,test_x_time.shape,train_x_force.shape,test_x_force.shape)
#    newli_train_time=labelprocess(train_y_time)
#    newli_test_time=labelprocess(test_y_time)
#    newli_train_force=labelprocess(train_y_force)
#    newli_test_force=labelprocess(test_y_force)

## 10 Fold cross validation
#    x_trains_force,y_trains_force,x_tests_force,y_tests_force = kfold_validation(a_force,b_force)
#    x_trains_time,y_trains_time,x_tests_time,y_tests_time = kfold_validation(a_time,b_time)
        
    
    # Parameters
    learning_rate = 0.001
    training_iters_force = 5000000
#    training_iters_time = 500000
    batch_size = 256
    display_step = 100
    batchid_time = 0
    batchid_force = 0
    
    # Network Parameters
    n_input_force = 15
    n_input_time = 32
    n_steps = 4
    n_hidden = 128

    # reset graph
    tf.reset_default_graph()

#   force_channel Graph
    G_force=tf.Graph()
    Sess_force=tf.Session(graph=G_force)
    with Sess_force.as_default(): 
        with G_force.as_default():
            with tf.variable_scope("force_channel") as scope:
                x_force = tf.placeholder("float", [None, n_steps, n_input_force],name='x_force')
                y_force  = tf.placeholder("float", [None, n_classes])
                weights = {
                    'weights_out_force': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='weights_out_force')
                }
                biases= {
                    'biases_out_force': tf.Variable(tf.random_normal([n_classes]),name='biases_out_force')
                }
                pred_force, out_force = RNN(x_force, weights['weights_out_force'],  biases['biases_out_force'], n_input_force)
                logits_scaled_force=tf.nn.softmax(pred_force)
                cost_force = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_force, labels=y_force))
                optimizer_force = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_force)
                correct_pred_force = tf.equal(tf.argmax(pred_force,1), tf.argmax(y_force,1))
                accuracy_force = tf.reduce_mean(tf.cast(correct_pred_force, tf.float32))
            Sess_force.run(tf.global_variables_initializer()) 
            saverf = tf.train.Saver()

#   time_channel Graph
    G_time=tf.Graph()
    Sess_time=tf.Session(graph=G_time)
    with Sess_time.as_default(): 
        with G_time.as_default():
            with tf.variable_scope("time_channel") as scope:
                x_time = tf.placeholder("float", [None, n_steps, n_input_time],name='x_time')
                y_time = tf.placeholder("float", [None, n_classes])
                weights = {
                    'weights_out_time': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='weights_out_time'),
                }
                biases= {
                    'biases_out_time': tf.Variable(tf.random_normal([n_classes]),name='biases_out_time'),
                }
                pred_time, out_time = RNN(x_time, weights['weights_out_time'],  biases['biases_out_time'], n_input_time)
                logits_scaled_time=tf.nn.softmax(pred_time)
                cost_time = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_time, labels=y_time))
                optimizer_time = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_time)
                correct_pred_time = tf.equal(tf.argmax(pred_time,1), tf.argmax(y_time,1))
                accuracy_time = tf.reduce_mean(tf.cast(correct_pred_time, tf.float32))
            Sess_time.run(tf.global_variables_initializer()) 
            savert = tf.train.Saver()
            
#    dcca_model Graph
    G_dcca=tf.Graph()
    Sess_dcca=tf.Session(graph=G_dcca)
    with Sess_dcca.as_default(): 
        with G_dcca.as_default():
            dcca_model=DCCA()
            input_view1 = dcca_model.input_view1
            input_view2 = dcca_model.input_view2
            hidden_view1 = dcca_model.output_view1
            hidden_view2 = dcca_model.output_view2
            hidden_view1_pred = dcca_model.output_view1_class
            hidden_view2_pred = dcca_model.output_view2_class
            label1 = dcca_model.label1
            label2 = dcca_model.label2
            neg_corr = dcca_model.neg_corr
            value= dcca_model.value
#            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#            Sess_dcca = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
            
#            maxmize the correlation between two data unsupervised learning(minimize -corr)
#            train_op = tf.train.MomentumOptimizer(learning_rate, 0.99).minimize(neg_corr,var_list=tf.trainable_variables())
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(neg_corr,var_list=tf.trainable_variables())
#            minimize the cost between different classes supervised learning
            cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label1, logits=hidden_view1_pred))
            optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy1)
            accuracy1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hidden_view1_pred, 1), tf.argmax(label1, 1)), tf.float32))
            
            cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label2, logits=hidden_view2_pred))
            optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy2)
            accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hidden_view2_pred, 1), tf.argmax(label2, 1)), tf.float32))
            
            lossfuse=cross_entropy1+cross_entropy2+tf.exp(neg_corr)
            optimizerfuse=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lossfuse)

##            supervised learning
#            cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=cnnlabel1, logits=hidden_view1))
#            optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy1)
#            cnnaccuracy1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hidden_view1, 1), tf.argmax(cnnlabel1, 1)), tf.float32))
#            
#            cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=cnnlabel2, logits=hidden_view2))
#            optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy2)
#            cnnaccuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hidden_view2, 1), tf.argmax(cnnlabel2, 1)), tf.float32))
            
            Sess_dcca.run(tf.global_variables_initializer()) 
            saverd = tf.train.Saver()
#            tf.InteractiveSession.close()
         


            
#    weights = {
#        'weights_out_time': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='weights_out_time'),
#        'weights_out_force': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='weights_out_force')
#    }
#    biases= {
#        'biases_out_time': tf.Variable(tf.random_normal([n_classes]),name='biases_out_time'),
#        'biases_out_force': tf.Variable(tf.random_normal([n_classes]),name='biases_out_force')
#    }
    
#    weights = {
#        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
#    }
#    biases= {
#        'out': tf.Variable(tf.random_normal([n_classes]))
#    }
    
    
#    
#    with tf.variable_scope("force_channel") as scope:
#        pred_force, out_force = RNN(x_force, weights['weights_out_force'], biases['biases_out_force'], n_input_force)
#        pred_force, out_force = RNN(x_force, weights['out'],  biases['out'], n_input_force)
#        logits_scaled_force=tf.nn.softmax(pred_force)
#        cost_force = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_force, labels=y_force))
#        optimizer_force = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_force)
#        correct_pred_force = tf.equal(tf.argmax(pred_force,1), tf.argmax(y_force,1))
#        accuracy_force = tf.reduce_mean(tf.cast(correct_pred_force, tf.float32))
#        
#    with tf.variable_scope("time_channel") as scope:
##        pred_time, out_time = RNN(x_time, weights['weights_out_time'],  biases['biases_out_time'], n_input_time)
#        pred_time, out_time = RNN(x_time, weights['out'],  biases['out'], n_input_time)
#        logits_scaled_time=tf.nn.softmax(pred_time)
#        cost_time = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_time, labels=y_time))
#        optimizer_time = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_time)
#        correct_pred_time = tf.equal(tf.argmax(pred_time,1), tf.argmax(y_time,1))
#        accuracy_time = tf.reduce_mean(tf.cast(correct_pred_time, tf.float32))
    
    
    

    accuracys_force=[]
    accuracys_time=[]
    
    for i in range(1):
        #20% split
        train_x_time,test_x_time,train_y_time,test_y_time = train_test_split(a_time,b_time,test_size=0.2,random_state=1)
        train_x_force,test_x_force,train_y_force,test_y_force = train_test_split(a_force,b_force,test_size=0.2,random_state=1)
        print(train_x_time.shape,test_x_time.shape,train_x_force.shape,test_x_force.shape)
        newli_train_time=labelprocess(train_y_time)
        newli_test_time=labelprocess(test_y_time)
        newli_train_force=labelprocess(train_y_force)
        newli_test_force=labelprocess(test_y_force)
        
        #10 fold            
#        train_x_force=x_trains_force[i]
#        train_y_force=y_trains_force[i]
#        test_x_force=x_tests_force[i]
#        test_y_force=y_tests_force[i]
#        
#        train_x_time=x_trains_time[i]
#        train_y_time=y_trains_time[i]
#        test_x_time=x_tests_time[i]
#        test_y_time=y_tests_time[i]
#        
#        newli_train_force=labelprocess(train_y_force)
#        newli_train_time=labelprocess(train_y_time)
#        
#        newli_test_force=labelprocess(test_y_force)
#        newli_test_time=labelprocess(test_y_time)


        # Initializing the variables
#        init = tf.global_variables_initializer()
#        saver = tf.train.Saver()
        
        # Launch the graph
#        with tf.Session() as sess:
#            #rnn
#            sess.run(init)
#            #rf
#        #    sess.run(rf_init_vars)
#            tf.device('/gpu:0')
        
        step = 1
        acc_forces=[]
        loss_forces=[]
        acc_times=[]
        loss_times=[]
        dccaloss=[]
        fuseloss=[]
        
        out_force256=None
        out_time256=None
        tf.device('/gpu:0')
        
        while step * batch_size < training_iters_force:
            with tf.variable_scope("force_channel") as scope:
                rf_batch_x_force, batch_y_force, rf_batch_y_force= next_batch(batch_size,train_x_force,train_y_force,newli_train_force,True)
                batch_x_force = rf_batch_x_force.reshape((batch_size, n_steps, n_input_force))
                _,out_force256=Sess_force.run([optimizer_force,out_force], 
                                              feed_dict={x_force: batch_x_force, y_force: batch_y_force})
                if step % display_step == 0:
                    acc_force,loss_force= Sess_force.run([accuracy_force,cost_force],
                                                         feed_dict={x_force: batch_x_force, y_force: batch_y_force})
                    print("Iter " + str(step*batch_size) + ", Minibatch loss_force= " + \
                      "{:.6f}".format(loss_force) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc_force)) 
                    acc_forces.append(acc_force)
                    loss_forces.append(loss_force)
                    
#                    step += 1    
#            step = 1
#            while step * batch_size < training_iters_time:  
            with tf.variable_scope("time_channel") as scope:
                rf_batch_x_time, batch_y_time, rf_batch_y_time= next_batch(batch_size,train_x_time,train_y_time,newli_train_time,False)
                batch_x_time = rf_batch_x_time.reshape((batch_size, n_steps, n_input_time))
                _,out_time256=Sess_time.run([optimizer_time,out_time], 
                                            feed_dict={x_time: batch_x_time, y_time: batch_y_time})
                if step % display_step == 0:
                    acc_time,loss_time = Sess_time.run([accuracy_time,cost_time], 
                                                       feed_dict={x_time: batch_x_time, y_time: batch_y_time})
                    print("Iter " + str(step*batch_size) + ", Minibatch loss_time= " + \
                      "{:.6f}".format(loss_time) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc_time)) 
                    acc_times.append(acc_time)
                    loss_times.append(loss_time)
                    
            ################# Deep CCA maxmize the correlation #############################
#            correlation in each node
#            for force256,time256 in zip(out_force256,out_time256):
#                _, neg_corr_val,_,_= Sess_dcca.run([train_op, neg_corr,optimizer1,optimizer2],
#                                                feed_dict={input_view1:force256,input_view2:time256,
#                                                           label1:batch_y_force,
#                                                           label2:batch_y_time})
#                acc1,acc2 = Sess_dcca.run([accuracy1, accuracy2],
#                                                feed_dict={input_view1:force256,input_view2:time256,
#                                                           label1:batch_y_force,
#                                                           label2:batch_y_time})
            
            for force256,time256 in zip(out_force256,out_time256):
#                print(force256.shape,time256.shape)
                _, neg_corr_val,_,lossfuseprint,corvalue= Sess_dcca.run([train_op, neg_corr,optimizerfuse,lossfuse,value],
                                                feed_dict={input_view1:force256,input_view2:time256,
                                                           label1:batch_y_force,
                                                           label2:batch_y_time})
                    
#                acc1,acc2 = Sess_dcca.run([accuracy1, accuracy2],
#                                                feed_dict={input_view1:force256,input_view2:time256,
#                                                           label1:batch_y_force,
#                                                           label2:batch_y_time})
#                print(corvalue)
            if step % display_step == 0:
                dccaloss.append(np.exp(neg_corr_val))
                fuseloss.append(lossfuseprint)
                    
#                print('corr_val',-neg_corr_val)
#                print("fuse_loss_for_train:", lossfuseprint)
#                print("accuracy1:", acc1)
#                print("accuracy2:", acc2)
                
                
            step += 1
                    
#           save the training process 
#            np.savetxt('./results/train_loss_dcca'+str(i)+'.csv',dccaloss,delimiter=',')
#            np.savetxt('./results/train_loss_fuse'+str(i)+'.csv',fuseloss,delimiter=',')
#            
#            
#            np.savetxt('./results/train_acc_force'+str(i)+'.csv',acc_forces,delimiter=',')
#            np.savetxt('./results/train_loss_force'+str(i)+'.csv',loss_forces,delimiter=',')
#            np.savetxt('./results/train_acc_time'+str(i)+'.csv',acc_times,delimiter=',')
#            np.savetxt('./results/train_loss_time'+str(i)+'.csv',loss_times,delimiter=',')


        ################# Linear CCA #############################
#        Using CCA to extract feature in each node in LSTM
        data_time=a_time.reshape((-1,n_steps, n_input_time))
        out256_time=Sess_time.run(out_time,feed_dict={x_time: data_time, y_time: all_fea_time})
        data_force=a_force.reshape((-1,n_steps, n_input_force))
        out256_force=Sess_force.run(out_force,feed_dict={x_force: data_force, y_force: all_fea_force})
        fusionfeature_data=np.c_[out256_time[-1],out256_force[-1]]  
        np.savetxt('./fusionfeature_Corrmnn_sdu.csv', fusionfeature_data, fmt='%.4f')
#        compute the correlation in each node in LSTM (timestep* batchsize * 256d)
        X1projlist=np.array([])
        X2projlist=np.array([])
        for eachnode_force,eachnode_time in zip(out256_force,out256_time):
            X1proj, X2proj = Sess_dcca.run([hidden_view1, hidden_view2],
                                      feed_dict={
                                    input_view1: eachnode_force,
                                    input_view2: eachnode_time})
#            (11340, 10) (756, 10)
            X1projlist=np.c_[X1projlist,X1proj] if X1projlist.size else X1proj 
            X2projlist=np.c_[X2projlist,X2proj] if X2projlist.size else X2proj
#        ccafuse_data,_ = feature_connect(X2projlist,X1projlist)
        ccafuse_data=np.c_[X2projlist,X1projlist]  
        print('----------ccafuse_data '+str(i)+'-----------')
#        (756, 1600) (756, 1600)
        np.savetxt('./ccafuse_sdu.csv', ccafuse_data, fmt='%.4f')
        
        
        
#            print("Linear CCA started!")
#            w = [None, None]
#            m = [None, None]
#            print(X1proj.shape, X2proj.shape)
#            w[0], w[1], m[0], m[1] = linear_cca(X1proj, X2proj, 10)
#            print("Linear CCA ended!")
#            X1proj -= m[0].reshape([1, -1]).repeat(len(X1proj), axis=0)
#            X1proj = np.dot(X1proj, w[0])
#            X1projlist=np.c_[X1projlist,X1proj] if X1projlist.size else X1proj 
#            print(X1projlist.shape)

        ################# testing LSTM #############################
        test_data=test_x_force.reshape((-1,n_steps, n_input_force))
        test_label=newli_test_force
        accuracy_force_out=Sess_force.run(accuracy_force, feed_dict={x_force: test_data, y_force: test_label})
        print("Force Testing Accuracy:",accuracy_force_out)   
        
        test_data=test_x_time.reshape((-1,n_steps, n_input_time))
        test_label=newli_test_time
        accuracy_time_out=Sess_time.run(accuracy_time, feed_dict={x_time: test_data, y_time: test_label})
        print("Time Testing Accuracy:",accuracy_time_out)
        
        accuracys_force.append(accuracy_force_out) 
        accuracys_time.append(accuracy_time_out)
            
            
    print(accuracys_force,accuracys_time)
    print('accuracys_force_mean:',np.mean(accuracys_force)) 
    print('accuracys_time_mean:',np.mean(accuracys_time))
    accuracys_force.append(np.mean(accuracys_force))
    accuracys_time.append(np.mean(accuracys_time))
    
#    np.savetxt('./test_result_fog.csv',[accuracys_force,accuracys_time])
            
            
##   extract the last output of the lstm in all data
#    data_time=a_time.reshape((-1,n_steps, n_input_time))
#    out256_time=Sess_time.run(out_time,feed_dict={x_time: data_time, y_time: all_fea_time})
#    
#    data_force=a_force.reshape((-1,n_steps, n_input_force))
#    out256_force=Sess_force.run(out_force,feed_dict={x_force: data_force, y_force: all_fea_force})
#
#    np.savetxt('./out256_time.txt', out256_time, fmt='%.4f')
#    np.savetxt('./out256_force.txt', out256_force, fmt='%.4f')
#        
#    saver.save(sess, './modelcache/fusemodel.ckpt')
#    writer=tf.summary.FileWriter('./fusemodel_graph',sess.graph)
#    writer.flush()
#    writer.close()
#    sess.close()
#    saverf.save(Sess_force, './modelcache/forcemodel.ckpt')
#    writerf=tf.summary.FileWriter('./graphs/forcemodel_graph',Sess_force.graph)
#    savert.save(Sess_time, './modelcache/timemodel.ckpt')
#    writert=tf.summary.FileWriter('./graphs/timemodel_graph',Sess_time.graph)     
#    saverd.save(Sess_dcca, './modelcache/dccamodel.ckpt')
#    writerd=tf.summary.FileWriter('./graphs/dccamodel_graph',Sess_dcca.graph)
#    writerf.flush()
#    writerf.close()
#    Sess_force.close()
#    writert.flush()
#    writert.close()
#    Sess_time.close()
#    writerd.flush()
#    writerd.close()
#    Sess_dcca.close()
    
    
#    align the two types of data 
#    fusionfeature_data,force_data = feature_connect(out256_time,out256_force)
#    fusionfeature_data=np.c_[out256_time[-1],out256_force[-1]]  
#    np.savetxt('./fusionfeature_Corrmnn.txt', fusionfeature_data, fmt='%.4f')
#    hmm_accuracy = hmm_4model_classification(fusionfeature_data,b_time)
    
    # combine the lda feature(2d) with ccafuse_data
#    ldafeature=np.loadtxt('./feature_extract/ldafeature_data.txt')
#    ldafeature=softmax(ldafeature)
#    ldafeature=preprocessing.normalize(ldafeature)
#    print(ldafeature)
#    ccafuse_data=np.c_[ccafuse_data,ldafeature]
#
#    hmm_accuracy = hmm_4model_classification(ccafuse_data,b_time)
#    print('Total hmm accuracy:',hmm_accuracy)



#    fuse_data=np.loadtxt('/home/zat/zresearch/ndds-corrlstm/results/fog/fusefea.csv')
#    
#    hmm_accuracy = hmm_3model_classification(fuse_data,b_time)
#    print('Total hmm accuracy:',hmm_accuracy)
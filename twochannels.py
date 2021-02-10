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
from hmmexm import hmm_4model_classification
from sklearn.model_selection import LeaveOneOut, KFold,cross_val_score
from deep_CCA_model import *
from linear_cca import linear_cca



def labelprocess(label,n_class=4):
    label_length=len(label)
    label_matrix=np.zeros((label_length,n_class))
    for i,j in enumerate(label): 
       label_matrix[i,n_class-int(j)]=1
    return label_matrix
    

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
    a_force=np.loadtxt("./txtfuse/force_aligndata_11340.txt")
    b_force=np.loadtxt("./txtfuse/force_alignlabel_11340.txt")
    
    a_time=np.loadtxt("./tsfuse/lrts10sdata12fea.txt")
    b_time=np.loadtxt("./tsfuse/lrts10slabel12fea.txt")
    
    all_fea_force=labelprocess(b_force)
    all_fea_time=labelprocess(b_time)
    
        
    
    # Parameters
    learning_rate = 0.001
    training_iters_force = 500000
#    training_iters_time = 500000
    batch_size = 256
    display_step = 100
    batchid_time = 0
    batchid_force = 0
    
    # Network Parameters
    n_input_force = 60
    n_input_time = 12
    n_steps = 10
    n_hidden = 256
    n_classes = 4
    
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
         


            

    accuracys_force=[]
    accuracys_time=[]
    
    for i in range(1):
        #20% split
        train_x_time,test_x_time,train_y_time,test_y_time = train_test_split(a_time,b_time,test_size=0.2)
        train_x_force,test_x_force,train_y_force,test_y_force = train_test_split(a_force,b_force,test_size=0.2)
        print(train_x_time.shape,test_x_time.shape,train_x_force.shape,test_x_force.shape)
        newli_train_time=labelprocess(train_y_time)
        newli_test_time=labelprocess(test_y_time)
        newli_train_force=labelprocess(train_y_force)
        newli_test_force=labelprocess(test_y_force)
        

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
                
                
            step += 1
                    
            np.savetxt('./results/train_acc_twochannel'+str(i)+'.csv',0.5*np.array(acc_forces)+0.5*np.array(acc_times), delimiter=',')
            np.savetxt('./results/train_loss_twochannel'+str(i)+'.csv',0.5*np.array(loss_forces)+0.5*np.array(loss_times), delimiter=',')
        
        
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
    
    np.savetxt('./results/test_result.csv',[accuracys_force,accuracys_time],delimiter=',')
            
            
##   extract the last output of the lstm in all data
#    data_time=a_time.reshape((-1,n_steps, 12))
#    out256_time=sess.run(out_time,feed_dict={x_time: data_time, y_time: all_fea_time})
#    
#    data_force=a_force.reshape((-1,n_steps, 60))
#    out256_force=sess.run(out_force,feed_dict={x_force: data_force, y_force: all_fea_force})
#
#    np.savetxt('./feature_extract/out256_time_10.txt', out256_time, fmt='%.4f')
#    np.savetxt('./feature_extract/out256_force_10.txt', out256_force, fmt='%.4f')
#        
#    saver.save(sess, './modelcache/fusemodel.ckpt')
#    writer=tf.summary.FileWriter('./fusemodel_graph',sess.graph)
#    writer.flush()
#    writer.close()
#    sess.close()
    saverf.save(Sess_force, './modelcache/forcemodel.ckpt')
    writerf=tf.summary.FileWriter('./graphs/forcemodel_graph',Sess_force.graph)
    savert.save(Sess_time, './modelcache/timemodel.ckpt')
    writert=tf.summary.FileWriter('./graphs/timemodel_graph',Sess_time.graph)     
    writerf.flush()
    writerf.close()
    Sess_force.close()
    writert.flush()
    writert.close()
    Sess_time.close()
    




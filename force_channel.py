#-*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:48:22 2018

@author: Aite Zhao
"""


from __future__ import print_function
import random
import tensorflow as tf 

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
#from tensorflow_hmm import hmm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
#fisher feature computation
#bayes model
n_classes = 4 # MNIST total classes (0-9 digits)


#rnd_clf = RandomForestClassifier(random_state=42)
## VotingClassifier
#hard_voting_clf = VotingClassifier(
#        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#        voting='hard'
#    )
#soft_voting_clf = VotingClassifier(
#        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
#        voting='soft'
#    )
#hard_voting_clf.fit(X_train, y_train)
#soft_voting_clf.fit(X_train, y_train)

 
#remove cpu occupation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
a=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/txtfuse/force_aligndata_11340.txt")
b=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/txtfuse/force_alignlabel_11340.txt")
print(a.shape)
b=b-1
#a=preprocessing.normalize(a) 
print(a.shape,b.shape)


#train_x=a
#train_y=b
train_x,test_x,train_y,test_y = train_test_split(a,b,test_size=0.2)
print(train_x.shape,test_x.shape)
m=len(train_y)
m1=len(test_y)
newli_train=np.zeros((m,n_classes))
newli_test=np.zeros((m1,n_classes))
for i,j in enumerate(train_y): 
   newli_train[i,int(j)]=1
#train_y=newli_train
for i,j in enumerate(test_y): 
   newli_test[i,int(j)]=1
#test_y=newli_test




#The HMM model for every output of each node in LSTM
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=10, cov_type='diag', n_iter=20):
        #模型名称 hmmlearn实现了三种HMM模型类，GaussianHMM和GMMHMM是连续观测状态的HMM模型，MultinomialHMM是离散观测状态的模型
        self.model=None
        self.model_name = model_name
        #隐藏状态个数
        self.n_components = n_components 
        #转移矩阵协方差类型
        self.cov_type = cov_type 
        #训练迭代次数
        self.n_iter = n_iter 
        self.models = []
        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
#            self.model = hmm.MultinomialHMM(n_components=self.n_components, n_iter=self.n_iter, tol=0.01)
            self.model = hmm.GMMHMM(n_components=self.n_components, n_iter=self.n_iter, tol=0.01)
#            raise TypeError('Invalid model type')

    # X是2维numpy数组每行13列
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # 测试输入的模型得分
    def get_score(self, input_data):
        return self.model.score(input_data)




       
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 2000000
batch_size = 256  
display_step = 200
batchid = 0


# Network Parameters
n_input = 60 #31  62 13MNIST data input (img shape: 28*28)
n_steps =  10 #50timesteps
n_hidden = 128# hidden layer num of features

#RF parameters
#num_steps = 3000 # Total steps to train
#num_features = 600 # Each image is 28x28 pixels
#num_trees = 300
#max_nodes = 3000
# 



# reset graph
tf.reset_default_graph()


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

#
##RF Input and Target data
#rf_x = tf.placeholder(tf.float32, shape=[None, num_features])
##RF For random forest, labels must be integers (the class id)
#rf_y = tf.placeholder(tf.int32, shape=[None])


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def next_batch(batch_size):
    global batchid 
    if batchid+batch_size > len(train_x):
       batchid = 0
    batch_data = (train_x[batchid:min(batchid +batch_size, len(newli_train)),:])
    batch_labels = (newli_train[batchid:min(batchid + batch_size, len(newli_train)),:])
    batch_labels_1d = (train_y[batchid:min(batchid + batch_size, len(train_y))])
    batchid = min(batchid + batch_size, len(newli_train))
    return batch_data, batch_labels,batch_labels_1d
    

#def RandomForest():
#    # Random Forest Parameters
#    hparams = tensor_forest.ForestHParams(num_classes=5,
#                                          num_features=num_features,
#                                          num_trees=num_trees,
#                                          max_nodes=max_nodes).fill()
#     
#    # Build the Random Forest
#    forest_graph = tensor_forest.RandomForestGraphs(hparams)
#    # Get training graph and loss
#    train_op = forest_graph.training_graph(rf_x, rf_y)
#    loss_op = forest_graph.training_loss(rf_x, rf_y)
#    
#    # Measure the accuracy
#    infer_op, tree_paths, regression_variance = forest_graph.inference_graph(rf_x)
#    
#    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(rf_y, tf.int64))
#    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    
#    # Initialize the variables (i.e. assign their default value) and forest resources
#    init_vars = tf.group(tf.global_variables_initializer(),resources.initialize_resources(resources.shared_resources()))
#    return train_op,loss_op,accuracy_op,init_vars,infer_op




def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
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
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell]*2)   
    # Get lstm cell output
    outputs, states = rnn.rnn(cell=lstm_cell, inputs=x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'],outputs[-1]



#pred batch*4d out batch*128d 
pred,out = RNN(x, weights, biases)

#RF return parameters
#rf_train_op,rf_loss_op,rf_accuracy_op,rf_init_vars,probabilities = RandomForest()
#Compare probabilities in LSTM and RF
#lstmpro= tf.reduce_max(pred,1)
#rfpro = tf.reduce_max(probabilities,1)
#mixpro = tf.equal(tf.argmax(pred,1),tf.argmax(probabilities,1)-1)
#Compare the predicted percentage of the two model, use the greater one as the label of the class
#finalpreresult=tf.where(tf.equal(tf.greater(lstmpro,rfpro),True),tf.argmax(pred,1),tf.argmax(probabilities,1)-1)
#preresult=tf.one_hot(finalpreresult,4,1,0)
    



# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=preresult, labels=y))
#第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，输出就是一个num_classes大小的向量
#logits_scaled=tf.nn.softmax(pred)
#第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵cross_entropy
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(knn_rnn_pred),reduction_indices=1))



#GWO optimizer
#optimizer = GWO.GWO(getattr(benchmarks, function_name),0,2,30,5,10000)
#Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#correct_pred = tf.equal(finalpreresult, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#rf Session
#sess = tf.train.MonitoredSession()


# Launch the graph
with tf.Session() as sess:
    #rnn
    sess.run(init)
    #rf
#    sess.run(rf_init_vars)
    tf.device('/gpu:0')
    step = 1
    # Training
    acclist=[]   
    losslist=[]
#    for i in range(1, num_steps + 1):
            # input out[-1] to rf classifier
#            batch_x, batch_y, rf_batch_y = next_batch(batch_size)
#            batch_x = batch_x.reshape((batch_size, n_steps, n_input)) 
#            out128data,acc,_,loss = sess.run([out,accuracy,optimizer,cost], feed_dict={x: batch_x, y: batch_y})
#            _, l, rf_acc = sess.run([rf_train_op, rf_loss_op,rf_accuracy_op], feed_dict={rf_x: out128data, rf_y: rf_batch_y})
#            if i % 200 == 0 or i == 1:
#                print('Step %i, Loss: %f, Acc: %f' % (i, l, rf_acc))
#                print("Iter " + str(i) + ", Minibatch Loss= " + \
#                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.5f}".format(acc))   
#    #    rf test  Accuracy
#    test_data = test_x.reshape((-1,n_steps, n_input))
#    test_label = newli_test
#    out128test,accuracy_test = sess.run([out,accuracy],feed_dict={x: test_data, y: test_label})
#    print("Testing Accuracy:", accuracy_test) 
#    print("RF Test Accuracy:", sess.run(rf_accuracy_op, feed_dict={rf_x: out128test, rf_y: test_y}))


        # heterogeneous ensemble learning LSTM+RF voting classification
        # Get the next batch of MNIST data (only images are needed, not labels)
#        rf_batch_x,_, rf_batch_y = next_batch(batch_size)
#        _, l = sess.run([rf_train_op, rf_loss_op], feed_dict={rf_x: rf_batch_x, rf_y: rf_batch_y})
#        if i % 200 == 0 or i == 1:
#            acc = sess.run(rf_accuracy_op, feed_dict={rf_x: rf_batch_x, rf_y: rf_batch_y})
#            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
#    #    rf test  Accuracy
#    print("RF Test Accuracy:", sess.run(rf_accuracy_op, feed_dict={rf_x: test_x, rf_y: test_y}))
        
        
        
        
  
#      Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        rf_batch_x, batch_y, rf_batch_y= next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = rf_batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#        sess.run([optimizer,rf_train_op, rf_loss_op,rf_accuracy_op], feed_dict={x: batch_x, y: batch_y,rf_x: rf_batch_x, rf_y: rf_batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc= sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            acclist.append(acc)
            
#            acc,_,_,_ = sess.run([accuracy,rf_train_op, rf_loss_op,rf_accuracy_op], feed_dict={x: batch_x, y: batch_y,rf_x: rf_batch_x, rf_y: rf_batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            losslist.append(loss)
            
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
#    
#    np.savetxt('acc_lstm.csv',acclist,delimiter=',')
#    np.savetxt('loss_lstm.csv',losslist,delimiter=',')
    # Calculate accuracy for 128 mnist test images
    test_data=test_x.reshape((-1,n_steps, n_input))
    test_label=newli_test
    predict_relt,ac=sess.run([pred,accuracy], feed_dict={x: test_data, y: test_label})
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    
    np.savetxt('test_label_ndds_'+str(round(ac,2))+'.csv',test_label)
    np.savetxt('predict_label_ndds_'+str(round(ac,2))+'.csv',predict_relt)
    
#        sess.run(accuracy, feed_dict={x: test_data, y: test_label,rf_x: test_x, rf_y: test_y}))
    
    saver.save(sess, './modelcache/forcemodel.ckpt')
    writer=tf.summary.FileWriter('./forcemodel_graph',sess.graph)
    writer.flush()
    writer.close()
    sess.close()
    
    #train_x = train_x.reshape((684, n_steps, n_input))
    #trainout=sess.run(out, feed_dict={x: train_x, y: train_y})
    #np.savetxt('./trainout.txt',trainout)
    # Calculate accuracy for 128 mnist test images
    #test_data=test_x.reshape((-1,n_steps, n_input))
    #test_label=test_y
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    #saver.save(sess, './model.ckpt')

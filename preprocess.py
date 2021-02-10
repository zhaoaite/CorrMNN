#-*- coding: utf-8 -*-

from __future__ import print_function
import random
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
#from tensorflow_hmm import hmm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut


a=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/sdugait/12023f.txt")
b=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/sdugait/12023label.txt")
#b=b-1
#a=preprocessing.normalize(a) 
print(a.shape,b.shape)
a1=np.load("/home/zat/zresearch/ndds-corrlstm/data/sdugait/docs.npy")
b1=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/sdugait/labels.txt")
b1=b1[:,1]
#a=preprocessing.normalize(a) 
print(a1.shape,b1.shape)
#a=preprocessing.normalize(a) 
sel=np.array([])
stamp=0

for i in range(52):
    count=0
    for index,j in enumerate(b):
        if i+1==j:
            count+=1
    sel=np.r_[sel,a1[stamp:stamp+count,:]] if sel.size else a1[stamp:stamp+count,:]
    stamp+=465
    print(stamp)
    
print(sel.shape)   
np.save('sdu.npy',sel)
    
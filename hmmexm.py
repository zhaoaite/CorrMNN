#-*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt 
# hmmlearn可以在安装numpy以后，再使用pip install hmmlearn安装 
from hmmlearn import hmm 
from sklearn.cross_validation import train_test_split
import os
from sklearn.model_selection import LeaveOneOut
import math
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import time
'''
Created on 2017-12-4
本例为天气和行为的关系
''' 
#states = ["Rainy", "Sunny"]
###隐藏状态 
#n_states = len(states)
###隐藏状态长度 
#observations = ["walk", "shop", "clean"]
###可观察的状态 
#n_observations = len(observations)
###可观察序列的长度 
#start_probability = np.array([0.6, 0.4])
###开始转移概率，即开始是Rainy和Sunny的概率 ##隐藏间天气转移混淆矩阵，即Rainy和Sunny之间的转换关系，例如[0,0]表示今天Rainy，明天Rainy的概率 
#transition_probability = np.array([[0.7, 0.3], [0.4, 0.6]]) 
###隐藏状态天气和可视行为混淆矩阵，例如[0,0]表示今天Rainy，walk行为的概率为0.1 
#emission_probability = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]) 
##构建了一个MultinomialHMM模型，这模型包括开始的转移概率，隐藏间天气转换混淆矩阵（transmat），隐藏状态天气和可视行为混淆矩阵emissionprob，对模型参数初始化 
#model = hmm.MultinomialHMM(n_components=n_states) 
#model.startprob_= start_probability 
#model.transmat_ = transition_probability 
#model.emissionprob_ = emission_probability 
##给出一个可见序列 
#bob_Actions = np.array([[2, 0, 1, 1, 2, 0]]).T 
## 解决问题1,解码问题,已知模型参数和X，估计最可能的Z;维特比算法 
#logprob, weathers = model.decode(bob_Actions, algorithm="viterbi")
#print("Bob Actions:",map(lambda x: observations[x], bob_Actions))
#print("weathers:",map(lambda x: states[x], weathers))
#print(logprob)
##该参数反映模型拟合的好坏,数值越大越好 # 解决问题2,概率问题，已知模型参数和X，估计X出现的概率, 向前-向后算法 
#score = model.score(bob_Actions, lengths=None) 
##最后输出结果 
#print(score)
#
#
#
#states = ["A", "B", "C"]
#n_states = len(states)
#
#observations = ["down","up"]
#n_observations = len(observations)
#
#p = np.array([0.7, 0.2, 0.1])
#a = np.array([
#  [0.5, 0.2, 0.3],
#  [0.3, 0.5, 0.2],
#  [0.2, 0.3, 0.5]
#])
#b = np.array([
#  [0.6, 0.2],
#  [0.3, 0.3],
#  [0.1, 0.5]
#])
#o = np.array([[1, 0, 1, 1, 1]]).T
#
#model = hmm.MultinomialHMM(n_components=n_states)
#model.startprob_= p
#model.transmat_= a
#model.emissionprob_= b
#
#logprob, h = model.decode(o, algorithm="viterbi")
#print("The hidden h", ", ".join(map(lambda x: states[x], h)))





class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=10, cov_type='diag', n_iter=100):
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
#        self.states = ["als", "control","hunt", "park"]
#        self.n_states = len(self.states)
        self.observations = []
        self.n_observations = len(self.observations)
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
    def get_score(self, input_data,label):
        return self.model.score(input_data,label)


def hmm_model_create(data,hmm_models,class_label,n_components=10):
    m,n=data.shape
    hmm_trainer = HMMTrainer() 
#    class_labels=class_label * np.ones((m,1))
    hmm_trainer.model.fit(data,class_label)
    startprob=1/n_components * np.ones((1, n_components)) 
    hmm_trainer.model.startprob_=startprob[0]
#    hmm_trainer.model.transmat_=  np.zeros((n_components, n_components)) 
#    for i in range(n_components-1):
#        hmm_trainer.model.transmat_[n_components-1,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
#    print(hmm_trainer.model.transmat_)
#    hmm_trainer.model.emissionprob_= B_emissionprob
    hmm_models.append((hmm_trainer, class_label)) 
    
def auto_data_split_fog(a,b):
    trainsets_x=[]
    testsets=[]
    
    train_x1,test_x1,train_y1,test_y1 = train_test_split(a[0:7650,:],b[0:7650],test_size=0.7)
    train_x2,test_x2,train_y2,test_y2 = train_test_split(a[7650:11880,:],b[7650:11880],test_size=0.7)
    train_x3,test_x3,train_y3,test_y3 = train_test_split(a[11880:,:],b[11880:],test_size=0.7)
    
    trainsets_x.append(train_x1)
    trainsets_x.append(train_x2)
    trainsets_x.append(train_x3)
    
    testsets_x=np.r_[test_x1,test_x2,test_x3]
    testsets_y=np.r_[test_y1,test_y2,test_y3]
    testsets.append(testsets_x)
    testsets.append(testsets_y)
    
    return trainsets_x, testsets

   
def auto_data_split(a,b):
    trainsets_x=[]
    testsets=[]
    
    train_x1,test_x1,train_y1,test_y1 = train_test_split(a[0:156,:],b[0:156],test_size=0.2)
    train_x2,test_x2,train_y2,test_y2 = train_test_split(a[156:348,:],b[156:348],test_size=0.2)
    train_x3,test_x3,train_y3,test_y3 = train_test_split(a[348:576,:],b[348:576],test_size=0.2)
    train_x4,test_x4,train_y4,test_y4 = train_test_split(a[576:756,:],b[576:756],test_size=0.2)
    
    trainsets_x.append(train_x1)
    trainsets_x.append(train_x2)
    trainsets_x.append(train_x3)
    trainsets_x.append(train_x4)
    
    testsets_x=np.r_[test_x1,test_x2,test_x3,test_x4]
    testsets_y=np.r_[test_y1,test_y2,test_y3,test_y4]
    testsets.append(testsets_x)
    testsets.append(testsets_y)
    
    return trainsets_x, testsets
    
    
def manul_data_split(a,b):

    trainset1=[]
    trainset2=[]
    trainset3=[]
    trainset4=[]
    trainset5=[]
    trainsets=[]
    testsets=[]
    
#    training data
#    training set 1
    trainset1.append(a[0:125,:])
    trainset1.append(a[156:310,:])
    trainset1.append(a[348:530,:])
    trainset1.append(a[576:720,:]) 
    trainsets.append(trainset1)
#    testing data 1
    test_x1=np.r_[a[125:156,:],a[310:348,:],a[530:576,:],a[720:756,:]]
    test_y1=np.r_[b[125:156],b[310:348],b[530:576],b[720:756]]   
    testsets.append([test_x1,test_y1])
    
#    training set 2
    trainset2.append(a[31:156,:])
    trainset2.append(a[156+38:348,:])
    trainset2.append(a[348+46:576,:])
    trainset2.append(a[576+36:756,:])
    trainsets.append(trainset2)
#    testing data 2
    test_x2=np.r_[a[0:31,:],a[156:156+38,:],a[348:348+46,:],a[576:576+36,:]]
    test_y2=np.r_[b[0:31],b[156:156+38],b[348:348+46],b[576:576+36]] 
    testsets.append([test_x2,test_y2])

#    training set 3
    trainset3.append(a[10:125+10,:])
    trainset3.append(a[156+10:310+10,:])
    trainset3.append(a[348+10:530+10,:])
    trainset3.append(a[576+10:720+10,:])
    trainsets.append(trainset3)
#    testing data 3
    test_x3=np.r_[a[0:10,:],a[135:156,:],a[156:166,:],a[320:348,:],a[348:358,:],a[540:576,:],a[576:586,:],a[730:756,:]]
    test_y3=np.r_[b[0:10],b[135:156],b[156:166],b[320:348],b[348:358],b[540:576],b[576:586],b[730:756]] 
    testsets.append([test_x3,test_y3])
    
#    training set 4
    trainset4.append(a[20:145,:])
    trainset4.append(a[176:330,:])
    trainset4.append(a[368:550,:])
    trainset4.append(a[596:740,:])
    trainsets.append(trainset4)
#    testing data 4
    test_x4=np.r_[a[0:20,:],a[145:156,:],a[156:176,:],a[330:348,:],a[348:368,:],a[550:576,:],a[576:596,:],a[740:756,:]]
    test_y4=np.r_[b[0:20],b[145:156],b[156:176],b[330:348],b[348:368],b[550:576],b[576:596],b[740:756]] 
    testsets.append([test_x4,test_y4])
    
#    training set 5
    trainset5.append(a[30:155,:])
    trainset5.append(a[186:340,:])
    trainset5.append(a[378:560,:])
    trainset5.append(a[606:750,:])
    trainsets.append(trainset5)
    
#    testing data 5
    test_x5=np.r_[a[0:30,:],a[145:156,:],a[156:176,:],a[330:348,:],a[348:368,:],a[550:576,:],a[576:596,:],a[740:756,:]]
    test_y5=np.r_[b[0:30],b[145:156],b[156:176],b[330:348],b[348:368],b[550:576],b[576:596],b[740:756]] 
    testsets.append([test_x5,test_y5])
    
    return trainsets, testsets
    
def softmax(x): 
    x_exp = np.exp(x) 
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) 
    s = x_exp / x_sum 
    return s

def hmm_3model_classification(a,b):
    # combine the lda feature(2d) with ccafuse_data
    accuracys=[]
#    hmm_temp = HMMTrainer()
#    trainsets, testsets = manul_data_split(a,b)
    print(a.shape)
    trainsets, testsets = auto_data_split_fog(a,b)
    hmm_models = []
    predictlabel=[]
    testlabel=[]
    
    for i in range(1):
        s_time=time.clock()
    #    4 HMM model
    #    als HMM
        hmm_model_create(trainsets[0],hmm_models,class_label=0)
    #    control HMM    
        hmm_model_create(trainsets[1],hmm_models,class_label=1)
    #    hunt HMM
        hmm_model_create(trainsets[2],hmm_models,class_label=2)
        
    #    20% as testing dataset
        test_x=testsets[0]
        test_y=testsets[1]
        count=0
        for i,j in enumerate(test_x):
            max_score = float('-inf')
            output_label = 0
            # 迭代所有模型 
            # 得分最高的模型对应的标签，即为输出标签（识别值）
            for item in hmm_models:
                hmm_model, label = item
                score = hmm_model.get_score(j.reshape(1, -1),np.array([label]))
        #            logprob, h = hmm_model.model.decode(j.reshape(-1, 1), algorithm="viterbi")
                if score > max_score:
                    max_score = score
                    output_label = label
                    
    #        # 打印输出
            predictlabel.append(output_label)
            testlabel.append(int(test_y[i]))

            if int(output_label)==int(test_y[i]):
                count=count+1
        accuracys.append(count/len(test_y))
        e_time=time.clock()
        np.savetxt('predict_hmm_label.csv',np.c_[testlabel, predictlabel],fmt='%.d')
#        cnf_matrix = confusion_matrix(testlabel, predictlabel)
#        np.set_printoptions(precision=4)
#        plt.figure()
#        #        cnf_matrix=np.array([[1,0.,0.,0.],[0.,0.9655,0.0345,0.],[0.,0.0166,0.9834,0.],[0.,0.0042,0.0112,0.9846]])
#        plot_confusion_matrix(cnf_matrix, classes=['2','2.5','3'], normalize=True,
#                          title='Normalized confusion matrix')
        plt.show()
        print(accuracys,e_time-s_time)
    return np.mean(accuracys)



#if __name__=='__main__':
def hmm_4model_classification(a,b):
    # combine the lda feature(2d) with ccafuse_data
#    ldafeature=np.loadtxt('./feature_extract/ldafeature_data.txt')
#    ldafeature=softmax(ldafeature)
#    print(ldafeature)
#    a=np.loadtxt("./feature_extract/ccafuse_data.txt")
#    b=np.loadtxt("./feature_extract/fusionfeature_label.txt")
#    a=np.c_[a,ldafeature]
    
#    a_force=np.loadtxt("/home/zat/zresearch/lstm/lstm_test_improve/feature_extract/out256_fc_10s_11340.txt")
#    b_force=np.loadtxt("/home/zat/zresearch/lstm/lstm_test_improve/feature_extract/force_alignlabel_11340.txt")
#    a_time=np.loadtxt("/home/zat/zresearch/lstm/lstm_test_improve/feature_extract/out256_ts_10s.txt")
#    B_emissionprob=np.loadtxt("/home/zat/zresearch/lstm/lstm_test_improve/feature_extract/aftersoftmax_force_10s_11340.txt")
#    train_x,test_x,train_y,test_y = train_test_split(a,b,test_size=0.2)
#    print(train_x.shape,test_x.shape)
    accuracys=[]
#    hmm_temp = HMMTrainer()
#    trainsets, testsets = manul_data_split(a,b)
    print(a.shape)
    trainsets, testsets = auto_data_split(a,b)
    hmm_models = []
    predictlabel=[]
    testlabel=[]
    
    for i in range(5):
        s_time=time.clock()
    #    4 HMM model
    #    als HMM
        hmm_model_create(trainsets[0],hmm_models,class_label=1)
    #    control HMM    
        hmm_model_create(trainsets[1],hmm_models,class_label=2)
    #    hunt HMM
        hmm_model_create(trainsets[2],hmm_models,class_label=3)
    #    parkinson HMM
        hmm_model_create(trainsets[3],hmm_models,class_label=4)
    #    20% as testing dataset
        test_x=testsets[0]
        test_y=testsets[1]
        count=0
        for i,j in enumerate(test_x):
            max_score = float('-inf')
            output_label = 0
            # 迭代所有模型 
            # 得分最高的模型对应的标签，即为输出标签（识别值）
            for item in hmm_models:
                hmm_model, label = item
                score = hmm_model.get_score(j.reshape(1, -1),np.array([label]))
        #            logprob, h = hmm_model.model.decode(j.reshape(-1, 1), algorithm="viterbi")
                if score > max_score:
                    max_score = score
                    output_label = label
                    
    #        # 打印输出
            predictlabel.append(output_label)
            testlabel.append(int(test_y[i]))

            if int(output_label)==int(test_y[i]):
                count=count+1
        accuracys.append(count/len(test_y))
        e_time=time.clock()
        cnf_matrix = confusion_matrix(testlabel, predictlabel)
        np.set_printoptions(precision=4)
        plt.figure()
        #        cnf_matrix=np.array([[1,0.,0.,0.],[0.,0.9655,0.0345,0.],[0.,0.0166,0.9834,0.],[0.,0.0042,0.0112,0.9846]])
        plot_confusion_matrix(cnf_matrix, classes=['CO','ALS','HD','PD'], normalize=True,
                          title='Normalized confusion matrix')
        plt.show()
        print(accuracys,e_time-s_time)
    return np.mean(accuracys)



    
#    hmm_trainer = HMMTrainer()
#    # 训练单个模型
#    hmm_trainer.model.fit(train_x)
#    # define the start hidden state 1,2,3,4 
##    startpro_state = np.array([13/63,16/63,19/63,15/63])
##    startpro_transmat = np.array([[0.99,0.01/3,0.01/3,0.01/3], [0.01/3,0.99,0.01/3,0.01/3],[0.01/3,0.01/3,0.99,0.01/3],[0.01/3,0.01/3,0.01/3,0.99]])
##    hmm_trainer.model.startprob_=startpro_state
##    hmm_trainer.model.transmat_= startpro_transmat
##    hmm_trainer.model.emissionprob_= B_emissionprob
##    print(B_emissionprob)
#    logprob, h = hmm_trainer.model.decode(test_x, algorithm="viterbi")
#   
#    for i,j in enumerate(h):
#        print("Predicted:", j+1,test_y[i])
#        if int(j+1)==int(test_y[i]):
#            count=count+1
#    print('accuracy:',count/len(test_y))
 
    
#    hmm_trainer = HMMTrainer(n_components=156)   
#    hmm_trainer.model.fit(a[0:156,:])
#    hmm_trainer.model.startprob_= 1/156 * np.ones((1, 156)) 
#    hmm_trainer.model.transmat_=  np.zeros((156, 156)) 
#    for i in range(155):
#        hmm_trainer.model.transmat_[155,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
#    print(hmm_trainer.model.transmat_)
#    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 1)) 
#    
#    
#    hmm_trainer = HMMTrainer(n_components=192)
#    hmm_trainer.model.fit(a[156:348,:])
#    hmm_trainer.model.startprob_=  1/192 * np.ones((1, 192)) 
#    hmm_trainer.model.transmat_= np.zeros((192, 192)) 
#    for i in range(191):
#        hmm_trainer.model.transmat_[191,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
#    print(hmm_trainer.model.transmat_)
#    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 2))  
#    
#    hmm_trainer = HMMTrainer(n_components=228)
#    hmm_trainer.model.fit(a[348:576,:])
#    hmm_trainer.model.startprob_=  1/228 * np.ones((1, 228)) 
#    hmm_trainer.model.transmat_= np.zeros((228, 228))
#    for i in range(227):
#        hmm_trainer.model.transmat_[227,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
#    print(hmm_trainer.model.transmat_)
#    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 3))  
#    
#    hmm_trainer = HMMTrainer(n_components=180)
#    hmm_trainer.model.fit(a[576:756,:])
#    hmm_trainer.model.startprob_= 1/180 * np.ones((1, 180)) 
#    hmm_trainer.model.transmat_= np.zeros((180, 180)) 
#    for i in range(179):
#        hmm_trainer.model.transmat_[179,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
#    print(hmm_trainer.model.transmat_)
#    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 4))



##    8 HMM models 
##    4 HMM time models
##    als HMM time
#    hmm_trainer = HMMTrainer(n_components=10)   
#    hmm_trainer.model.fit(a_time[0:125,:])
#    startprob=1/10 * np.ones((1, 10)) 
#    hmm_trainer.model.startprob_=startprob[0] 
#    hmm_trainer.model.transmat_=  np.zeros((10, 10)) 
#    for i in range(9):
#        hmm_trainer.model.transmat_[9,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
##    print(hmm_trainer.model.transmat_)
##    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 1)) 
#    
##    control HMM time 
#    hmm_trainer = HMMTrainer(n_components=10)
#    hmm_trainer.model.fit(a_time[156:310,:])
#    startprob=1/10 * np.ones((1, 10)) 
#    hmm_trainer.model.startprob_=startprob[0] 
#    hmm_trainer.model.transmat_= np.zeros((10, 10)) 
#    for i in range(9):
#        hmm_trainer.model.transmat_[9,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1    
##    print(hmm_trainer.model.transmat_)
##    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 2))
#    
##    hunt HMM time    
#    hmm_trainer = HMMTrainer(n_components=10)
#    hmm_trainer.model.fit(a_time[348:530,:])
#    startprob=1/10 * np.ones((1, 10)) 
#    hmm_trainer.model.startprob_=startprob[0] 
#    hmm_trainer.model.transmat_= np.zeros((10, 10)) 
#    for i in range(9):
#        hmm_trainer.model.transmat_[9,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
##    print(hmm_trainer.model.transmat_)
##    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 3))  
#    
##    parkinson HMM time  
#    hmm_trainer = HMMTrainer(n_components=10)
#    hmm_trainer.model.fit(a_time[576:720,:])
#    startprob=1/10 * np.ones((1, 10)) 
#    hmm_trainer.model.startprob_=startprob[0]     
#    hmm_trainer.model.transmat_= np.zeros((10, 10)) 
#    for i in range(9):
#        hmm_trainer.model.transmat_[9,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1    
##    print(hmm_trainer.model.transmat_)
##    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 4))  
#    
#
#
##    4 HMM force models
##    als HMM force
#    hmm_trainer = HMMTrainer(n_components=10)   
#    hmm_trainer.model.fit(a_force[0:1872,:])
#    startprob=1/10 * np.ones((1, 10)) 
#    hmm_trainer.model.startprob_=startprob[0] 
#    hmm_trainer.model.transmat_=  np.zeros((10, 10)) 
#    for i in range(9):
#        hmm_trainer.model.transmat_[9,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
##    print(hmm_trainer.model.transmat_)
##    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 1)) 
#    
##    control HMM force 
#    hmm_trainer = HMMTrainer(n_components=10)
#    hmm_trainer.model.fit(a_force[2340:4644,:])
#    startprob=1/10 * np.ones((1, 10)) 
#    hmm_trainer.model.startprob_=startprob[0] 
#    hmm_trainer.model.transmat_=  np.zeros((10, 10)) 
#    for i in range(9):
#        hmm_trainer.model.transmat_[9,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1    
##    print(hmm_trainer.model.transmat_)
##    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 2))
#    
##    hunt HMM force    
#    hmm_trainer = HMMTrainer(n_components=10)
#    hmm_trainer.model.fit(a_force[5220:7956,:])
#    startprob=1/10 * np.ones((1, 10)) 
#    hmm_trainer.model.startprob_=startprob[0] 
#    hmm_trainer.model.transmat_=  np.zeros((10, 10)) 
#    for i in range(9):
#        hmm_trainer.model.transmat_[9,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1
##    print(hmm_trainer.model.transmat_)
##    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 3))  
#    
##    parkinson HMM force  
#    hmm_trainer = HMMTrainer(n_components=10)
#    hmm_trainer.model.fit(a_force[8640:10800,:])
#    startprob=1/10 * np.ones((1, 10)) 
#    hmm_trainer.model.startprob_=startprob[0]     
#    hmm_trainer.model.transmat_=  np.zeros((10, 10)) 
#    for i in range(9):
#        hmm_trainer.model.transmat_[9,0]=1
#        hmm_trainer.model.transmat_[i,i+1] = 1    
##    print(hmm_trainer.model.transmat_)
##    hmm_trainer.model.emissionprob_= B_emissionprob
#    hmm_models.append((hmm_trainer, 4))     
#
#
#
##    20% as testing dataset
#    test_x=np.r_[a_time[125:156,:],a_time[310:348,:],a_time[530:576,:],a_time[720:756,:],a_force[1872:2340,:],a_force[4644:5220,:],a_force[7956:8640,:],a_force[10800:11340,:]]
#    test_y=np.r_[b[125:156],b[310:348],b[530:576],b[720:756],b_force[1872:2340],b_force[4644:5220],b_force[7956:8640],b_force[10800:11340]]
#    
#    train_x,test_x,train_y,test_y=train_test_split(test_x,test_y,test_size=0.99)
#
##    20% as testing dataset
#    for i,j in enumerate(test_x):
#        print(j.shape)
#        max_score = float('-inf')
#        output_label = 0
#        for item in hmm_models:
#            hmm_model, label = item
#            score = hmm_model.get_score(j.reshape(1, -1))
#            if score > max_score:
#                max_score = score
#                output_label = label
#
#        print("Predicted:", output_label,test_y[i])
#        if int(output_label)==int(test_y[i]):
#            count=count+1
#    print('accuracy:',count/len(test_y))

  
#    # regard each sample as an HMM model
#    for i,j in enumerate(train_x):
#        # 初始化变量 
#        X = np.array([])
#        y_words = []
#        X=j.reshape(-1, 1)
#        # 添加标签
#        y_words=train_y
##        print('X.shape =', X.shape)
#        # 训练并保存模型
#        
##        state 1,2,3,4 define the hidden state
##        startpro_state = np.random.uniform(0, 0.0000000001, (1, 4))
##        startpro_state = np.zeros((1, 4))
##        startpro_state[0][int(train_y[i]-1.0)]=1
##        hmm_trainer.model.transmat_= np.array([[1.0,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]) 
##        print(hmm_trainer.model.startprob_,train_y[i])
##        print(hmm_trainer.model.transmat_)
#       
#        max_score = float('-inf')
#        for c in range(1):
#            # 创建HMM类 
#            hmm_trainer = HMMTrainer()
#            # 训练模型
#            hmm_trainer.train(X)
#            # define the start hidden state 1,2,3,4 
##            startpro_state = np.array([13/63,16/63,19/63,15/63])
##            startpro_transmat = np.array([[0.99,0.01/3,0.01/3,0.01/3], [0.01/3,0.99,0.01/3,0.01/3],[0.01/3,0.01/3,0.99,0.01/3],[0.01/3,0.01/3,0.01/3,0.99]])
##            hmm_trainer.model.startprob_=startpro_state
##            hmm_trainer.model.transmat_= startpro_transmat
##            hmm_trainer.model.emissionprob_= B_emissionprob
#            score = hmm_trainer.get_score(X)
#            if score > max_score:
#                max_score = score
#                hmm_temp=hmm_trainer
#        hmm_models.append((hmm_temp, train_y[i]))
#        hmm_trainer = None
#    
#    # 测试文件路径
#    # 读取测试文件
#    for i,j in enumerate(test_x):
#        # 读取测试文件
#        # 定义模型得分，输出标签
#        max_score = float('-inf')
#        output_label = 0
#        # 迭代所有模型 
#        # 得分最高的模型对应的标签，即为输出标签（识别值）
#        for item in hmm_models:
#            hmm_model, label = item
#            score = hmm_model.get_score(j.reshape(-1, 1))
#            logprob, h = hmm_model.model.decode(j.reshape(-1, 1), algorithm="viterbi")
##            print(h)
#            if score > max_score:
#                max_score = score
#                output_label = label
#    
#        # 打印输出
#        print("Predicted:", output_label,test_y[i])
#        if int(output_label)==int(test_y[i]):
#            count=count+1
#    print('accuracy:',count/len(test_y))
#            
            








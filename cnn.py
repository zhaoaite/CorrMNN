#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import os
from sklearn import preprocessing


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

batchid=0
n_input = 30
n_steps = 20
n_classes = 4


def labelprocess(label,n_class=n_classes):
    label_length=len(label)
    label_matrix=np.zeros((label_length,n_class))
    for i,j in enumerate(label): 
       label_matrix[i,int(j)]=1
    return label_matrix

def next_batch(batch_size,train_x,train_y):
    global batchid
    if batchid+batch_size > len(train_x):
       batchid = 0
    batch_data = (train_x[batchid:min(batchid+batch_size, len(train_y)),:])
    batch_labels = (train_y[batchid:min(batchid + batch_size, len(train_y))])
    batchid = min(batchid + batch_size, len(train_y))
    return batch_data, batch_labels


batch_size=16
a=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/txtfuse/force_aligndata_11340.txt")
b=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/data/txtfuse/force_alignlabel_11340.txt")
print(a.shape)
b=b-1



train_x,test_x,train_y,test_y = train_test_split(a,b,test_size=0.2)
label_train= labelprocess(train_y)
label_test= labelprocess(test_y)

#one_hot is encoding format
#None means tensor 的第一维度可以是任意维度
#/255. 做均一化
#input_x=tf.placeholder(tf.float32,[None,n_input*n_steps])/(n_input*n_steps)
input_x=tf.placeholder(tf.float32,[None,n_input*n_steps])
#输出是一个one hot的向量
output_y=tf.placeholder(tf.int32,[None,n_classes])

#输入层 [28*28*1]
input_x_images=tf.reshape(input_x,[-1,n_input,n_steps,1])





conv1=tf.layers.conv2d(
    inputs=input_x_images,
    filters=32,
    kernel_size=[5,5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print(conv1)

#输出变成了 [28*28*32]

#pooling layer1 2*2
#tf.layers.max_pooling2d
#inputs 输入，张量必须要有四个维度
#pool_size: 过滤器的尺寸

pool1=tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2,2],
    strides=2
)
print(pool1)
#输出变成了[?,14,14,32]

#conv2 5*5*64
conv2=tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5,5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)

#输出变成了  [?,14,14,64]

#pool2 2*2
pool2=tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=2
)

#输出变成了[?,7,7,64]
print(pool2)
#flat(平坦化)
flat=tf.reshape(pool2,[-1,7*5*64])


#形状变成了[?,3136]

#densely-connected layers 全连接层 1024
#tf.layers.dense
#inputs: 张量
#units： 神经元的个数
#activation: 激活函数
dense=tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)

#输出变成了[?,1024]
print(dense)

dropout=tf.layers.dropout(
    inputs=dense,
    rate=0.5,
)
print(dropout)

#输出层，不用激活函数（本质就是一个全连接层）
logits=tf.layers.dense(
    inputs=dropout,
    units=n_classes
)
#输出形状[?,10]
print(logits)

#计算误差 cross entropy（交叉熵），再用Softmax计算百分比的概率
#tf.losses.softmax_cross_entropy
#onehot_labels: 标签值
#logits: 神经网络的输出值
loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                     logits=logits)
# 用Adam 优化器来最小化误差,学习率0.001 类似梯度下降
print(loss)
train_op=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


#精度。计算预测值和实际标签的匹配程度
#tf.metrics.accuracy
#labels：真实标签
#predictions: 预测值
#Return: (accuracy,update_op)accuracy 是一个张量准确率，update_op 是一个op可以求出精度。
#这两个都是局部变量
accuracy_op=tf.metrics.accuracy(
    labels=tf.argmax(output_y,axis=1),
    predictions=tf.argmax(logits,axis=1)
)[1] #为什么是1 是因为，我们这里不是要准确率这个数字。而是要得到一个op

#创建会话
sess=tf.Session()
#初始化变量
#group 把很多个操作弄成一个组
#初始化变量，全局，和局部
init=tf.group(tf.global_variables_initializer(),
              tf.local_variables_initializer())
sess.run(init)

for i in range(10000):
    batch_x, batch_y = next_batch(batch_size,train_x,train_y)
    batch_y_1 = labelprocess(batch_y)
    train_loss,train_op_=sess.run([loss,train_op],{input_x:batch_x,output_y:batch_y_1})
    if i%100==0:
        test_accuracy=sess.run(accuracy_op,{input_x:test_x,output_y:label_test})
        print("Step=%d, Train loss=%.4f,[Test accuracy=%.2f]"%(i,train_loss,test_accuracy))

#测试： 打印20个预测值和真实值 对
test_output=sess.run(logits,{input_x:test_x})
test_accuracy=sess.run(accuracy_op,{input_x:test_x,output_y:label_test})

print("------",test_accuracy)
np.savetxt('ndds_predict_cnn.csv',np.argmax(test_output,1))
np.savetxt('ndds_true_cnn.csv',label_test)
#inferenced_y=np.argmax(test_output,1)
#print(inferenced_y,'Inferenced numbers')#推测的数字
#print(np.argmax(test_y,1),'Real numbers')
sess.close()

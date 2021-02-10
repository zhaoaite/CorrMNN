#!/usr/bin/python
# -*- coding: utf-8 -*- 
import math
import numpy as np
from matplotlib import pyplot as plt  
from fishervector import FisherVectorGMM
from LDAclassifier import lda_reduse_dimension
 

import time
# fisher feature
def test_with_gaussian_samples_image(data,n_kernels):
    test_data = np.array(data)
    test_data = test_data.reshape([test_data.shape[0],-1,1])
    print(test_data.shape)
    fv_gmm = FisherVectorGMM(n_kernels=n_kernels).fit(test_data)
    n_test_videos = len(data)
    fv = fv_gmm.predict(test_data[:n_test_videos])
    print(fv.shape)
    return fv


# frequncy
def FFT(a):
    transy=np.fft.fft(a)  
#    plt.subplot(311),plt.plot(a),plt.title("Original")  
#    plt.subplot(312),plt.plot(transy),plt.title("FFT")  
    return transy

# time space
def  psfeatureTime(data,p1,p2):
    #均值
    df_mean=data[p1:p2].mean()
    #方差
    df_var=data[p1:p2].var()
    #标准差
    df_std=data[p1:p2].std()
    #均方根
    df_rms=math.sqrt(pow(df_mean,2) + pow(df_std,2))
    #偏度
    df_skew = np.mean((data[p1:p2] - df_mean) ** 3)
    #峭度
    df_kurt=np.mean((data[p1:p2]  - df_mean) ** 4) / pow(df_var, 2) 
    sum=0
    for i in range(p1,p2):
        sum+=math.sqrt(abs(data[i]))
    #波形因子
    df_boxing=df_rms / (abs(data[p1:p2]).mean())
    #峰值因子
    df_fengzhi=(max(data[p1:p2])) / df_rms
    #脉冲因子
    df_maichong=(max(data[p1:p2])) / (abs(data[p1:p2]).mean())
    #裕度因子
    df_yudu=(max(data[p1:p2])) / pow((sum/(p2-p1)),2)
    featuretime_list = [df_mean,df_rms,df_skew,df_kurt,df_boxing,df_fengzhi,df_maichong,df_yudu]
    return featuretime_list 

if __name__=='__main__':
#   load data 
    a_force=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/dataset_fog_release/dataset/HY/ga-align.csv")
    a_time=np.loadtxt("/home/zat/zresearch/ndds-corrlstm/dataset_fog_release/dataset/HY/fogdata.csv")
    m,n=a_force.shape
#    fftforce=[]
#    pstf_list=[]
    s_time=time.clock()
#    for i in range(m):
##        FFT feature
#        fftforce.append(FFT(a_force[i,:]).real)
##        time feature
#        timespace=psfeatureTime(a_force[i,:],0,len(a_force[i,:]))
#        pstf_list.append(timespace)
##    np.savetxt('./feature_extract/timespace_data.txt', pstf_list, fmt='%.4f')
##    np.savetxt('./feature_extract/fftfeature_data.txt', fftforce, fmt='%.4f')
##    plt.subplot(313),plt.plot(fftforce),plt.title("fuse")  
##    fisher_time=test_with_gaussian_samples_image(pstf_list) 
#    fftforce=np.array(fftforce)
#    pstf_list=np.array(pstf_list)
    
    
#    combine the two domains
#    a_force1=np.c_[fftforce,pstf_list]
    fisher_force=test_with_gaussian_samples_image(a_force,1)
    fisher_time=test_with_gaussian_samples_image(a_time,1)
    print(fisher_force.shape,fisher_time.shape)
    fusion_feature=np.c_[fisher_force.reshape(17811,-1),fisher_time.reshape(17811,-1)]
    
#    fisher_fft=test_with_gaussian_samples_image(a_force,10)
#    fisher_time=test_with_gaussian_samples_image(a_time,5)
#    fusion_feature=np.c_[pstf_list.reshape(pstf_list.shape[0],-1),fisher_fft.reshape(fisher_fft.shape[0],-1)]
#    fusion_feature=fusion_feature.reshape(756,-1)
#    fusion_feature=np.c_[fisher_time.reshape(756,-1),fusion_feature]
    
#    lda_reduse_dimension(fusion_feature)
    e_time=time.clock()
    print(e_time-s_time)
    np.savetxt('./fisher_feature_fog.txt', fusion_feature, fmt='%.8f')

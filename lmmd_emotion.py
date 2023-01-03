#!/usr/bin/env python
# encoding: utf-8
import torch
from Weight_emotion import Weight

'''
原文说法（译）：为了实现正确的对齐，我们设计了一个局部最大平均偏差(LMMD)，它度量了在考虑不同样本权重的情况下，
源域相关子域经验分布核均值嵌入 与 目标域核均值嵌入之间的Hilbert Schmidt范数。
'''

# 高斯核函数
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵
    参数: 
     source: 源域数据，行表示样本数目，列表示样本数据维度
     target: 目标域数据 同source
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
  sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])  
    # 求矩阵的行数，即两个域的的样本总数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0) 
    #将source,target按列方向合并，即合并后列数不变，行数增加
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total复制（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    L2_distance = ((total0-total1)**2).sum(2)
    # total0 - total1 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差 
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方

    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    # 多核MMD
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list] #高斯核函数的数学表达式
    return sum(kernel_val) #/len(kernel_val) #得到最终的核矩阵
                                                                                                         

#LMMD（局部最大均值误差）来度量子域的核均值之间的差异
def lmmd_emotion(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    
    weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, type='visual')
    
    weight_ss = torch.from_numpy(weight_ss).cuda() 
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    
    if torch.sum(torch.isnan(sum(kernels))):  #torch.isnan()来检查tensor数据中是否存在脏数据
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    #loss += torch.sum( SS + TT - 2 * ST )
    loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
    
    return loss

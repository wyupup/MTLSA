# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 18:24:28 2021

@author: ZHUANG
"""

import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from lmmd_emotion import lmmd_emotion
from lmmd_gender import lmmd_gender
import network as models
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

lr=0.000001
batch_size = 100   
epochs = 300
hidden_size = 256
class_emotion = 5
class_gender = 2

class MyDataset1(Dataset):
    def __init__(self, data_path, emotion_label_path, gender_label_path):
        self.data = np.load(data_path)
        self.emotion_label = np.load(emotion_label_path)
        self.gender_label = np.load(gender_label_path)
 
    def __getitem__(self, index):
        data = self.data[index]
        emotion_label = self.emotion_label[index]
        gender_label = self.gender_label[index]
        return data, emotion_label, gender_label
 
    def __len__(self):
        return len(self.emotion_label)

class MyDataset2(Dataset):
    def __init__(self, data_path, emotion_label_path):
        self.data = np.load(data_path)
        self.emotion_label = np.load(emotion_label_path)
 
    def __getitem__(self, index):
        data = self.data[index]
        emotion_label = self.emotion_label[index]
        return data, emotion_label
 
    def __len__(self):
        return len(self.emotion_label)

#1582维
#E:/cross_corpus_database/B-E/IS10_feature/Berlin_IS10_375_disgust.npy
#E:/cross_corpus_database/B-E/label_Berlin_disgust.npy
#E:/cross_corpus_database/B-E/gender_label_Berlin.npy
#E:/cross_corpus_database/B-E/IS10_feature/eNTERFACE_IS10_1072_disgust.npy
#E:/cross_corpus_database/B-E/label_eNTREFACE_disgust.npy
#E:/cross_corpus_database/B-E/gender_label_eNTREFACE.npy

#E:/cross_corpus_database/B-C/IS10_feature/Berlin_IS10_408_neutral.npy
#E:/cross_corpus_database/B-C/label_Berlin_neutral.npy
#E:/cross_corpus_database/B-C/gender_label_Berlin.npy
#E:/cross_corpus_database/B-C/IS10_feature/CASIA_IS10_1000_neutral.npy
#E:/cross_corpus_database/B-C/label_CASIA_neutral.npy
#E:/cross_corpus_database/B-C/gender_label_CASIA.npy

#E:/cross_corpus_database/C-E/IS10_feature/CASIA_IS10_1000_surprise.npy
#E:/cross_corpus_database/C-E/label_CASIA_surprise.npy
#E:/cross_corpus_database/C-E/gender_label_CASIA.npy
#E:/cross_corpus_database/C-E/IS10_feature/eNTERFACE_IS10_1072_surprise.npy
#E:/cross_corpus_database/C-E/label_eNTREFACE_surprise.npy
#E:/cross_corpus_database/C-E/gender_label_eNTREFACE.npy

def load_training():        
    data_path = 'E:/cross_corpus_database/C-E/IS10_feature/eNTERFACE_IS10_1072_surprise.npy'
    emotion_label_path = 'E:/cross_corpus_database/C-E/label_eNTREFACE_surprise.npy'
    gender_label_path = 'E:/cross_corpus_database/C-E/gender_label_eNTREFACE.npy'
    dataset = MyDataset1(data_path,emotion_label_path,gender_label_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)  
    return train_loader

def load_testing():        
    data_path = 'E:/cross_corpus_database/C-E/IS10_feature/CASIA_IS10_1000_surprise.npy'
    emotion_label_path = 'E:/cross_corpus_database/C-E/label_CASIA_surprise.npy'
    dataset = MyDataset2(data_path,emotion_label_path)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)  
    return test_loader

source_loader = load_training()
target_loader = load_testing()

len_target_dataset = len(target_loader.dataset)
len_target_loader = len(target_loader) 
len_source_loader = len(source_loader)

def train(epoch, model): 
    optimizer = torch.optim.Adam([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_emotion.parameters(),'lr': lr},
            {'params': model.cls_gender.parameters(),'lr': lr},
            ], lr=lr, weight_decay=5e-4)
    #nn.MSELoss() #均方误差损失函数,无限制
    #nn.BCELoss() #解码层最后需用sigmoid激活函数
    Loss1 = nn.BCELoss() 
    Loss2 = nn.BCELoss()
      
    model.train()    
    iter_source = iter(source_loader) #iter()用来生成一个迭代器
    iter_target = iter(target_loader)
        
    #num_iter = len_source_loader
    num_iter = len_target_loader
    
    train_loss = 0
    for i in range(0, num_iter):           
        data_source, emotion_label_source, gender_label_source = iter_source.next() #得到源域数据和标签
        data_target, label_target = iter_target.next() #目标域训练数据 #目标域标签是没有使用的
        
        s_e_label = emotion_label_source
        s_g_label = gender_label_source
        emotion_label_source = emotion_label_source.clone().detach().long()
        gender_label_source = gender_label_source.clone().detach().long()
      
        #归一化
        min_max_scaler = MinMaxScaler()  
        data_source = min_max_scaler.fit_transform(data_source)
        data_target = min_max_scaler.fit_transform(data_target)
        data_source = torch.Tensor(data_source)  
        data_target = torch.Tensor(data_target)          
        
        #if i % len_target_loader == 0:
            #iter_target = iter(target_loader)                                   
        
        if i % len_source_loader == 0:
            iter_source = iter(source_loader)
            
        '加噪声'
        noise = torch.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(1582,)))        
        data_noise_source = data_source + 0.6*noise
        data_noise_target = data_target + 0.6*noise
        
        if torch.cuda.is_available():           
            data_noise_source = data_noise_source.cuda()
            data_noise_target = data_noise_target.cuda()
            data_source = data_source.cuda()
            data_target = data_target.cuda()
            emotion_label_source = emotion_label_source.cuda()
            gender_label_source = gender_label_source.cuda()
        data_noise_source = Variable(data_noise_source)
        data_noise_target = Variable(data_noise_target)
        data_source = Variable(data_source)
        emotion_label_source = Variable(emotion_label_source)
        gender_label_source = Variable(gender_label_source)
        data_target = Variable(data_target)       
        
        optimizer.zero_grad()                 
        
        out = model(data_noise_source, data_noise_target)
        source_decoder_out, target_decoder_out = out[1],out[5]
        s_encoder_out, t_encoder_out, s_e_p, s_g_p, t_e_p, t_g_p = out[0], out[4], out[2], out[3], out[6], out[7] 
                                            
              
        pred_label = torch.nn.functional.log_softmax(s_e_p, dim=1)
        loss_cls_emotion = torch.nn.functional.nll_loss(pred_label, emotion_label_source)
        
        pred_label = torch.nn.functional.log_softmax(s_g_p, dim=1)
        loss_cls_gender = torch.nn.functional.nll_loss(pred_label, gender_label_source)
        
        t_e_label = torch.nn.functional.softmax(t_e_p, dim=1)            
        t_g_label = torch.nn.functional.softmax(t_g_p, dim=1)
        
        loss_lmmd_emotion = lmmd_emotion(s_encoder_out, t_encoder_out, s_e_label, t_e_label) 
        
        loss_lmmd_gender = lmmd_gender(s_encoder_out, t_encoder_out, s_g_label, t_g_label)
       
        loss1 = Loss1(source_decoder_out, data_source)
        loss2 = Loss2(target_decoder_out, data_target)
     
        loss = 0.05*loss1 + 0.05*loss2 + 0.6*loss_cls_emotion + 0.1*loss_cls_gender + 0.1*loss_lmmd_emotion + 0.1*loss_lmmd_gender 
        #loss = 0.05*loss1 + 0.05*loss2 + 0.8*loss_lmmd_emotion + 0.1*loss_lmmd_gender 
        train_loss += loss
                
        loss.backward()
        optimizer.step()
        #print('Train Epoch: {}\tLoss: {:.2f}\tsoft_Loss: {:.2f}\tmmd_Loss: {:.2f}\tlmmd_Loss: {:.4f}'.format(
                #epoch, loss.item(), loss_cls.item(), loss_mmd.item(),loss_lmmd.item()))                
        
    train_loss /= num_iter
    print('Train Epoch: {}\ttrain loss: {:.2f}'.format(epoch, train_loss.item()))   
  
def test(model):
    model.eval()
    with torch.no_grad():
        x_test = np.load('E:/cross_corpus_database/C-E/IS10_feature/CASIA_IS10_1000_surprise.npy')
        y_test = np.load('E:/cross_corpus_database/C-E/label_CASIA_surprise.npy')
  
        min_max_scaler = MinMaxScaler()   
        x_test = min_max_scaler.fit_transform(x_test)
        x_test = torch.Tensor(x_test).cuda()
        y_test = torch.Tensor(y_test).cuda().long()    
        x_test = Variable(x_test)
        y_test = Variable(y_test) 
        
        out = model(x_test, x_test)
        s_output= out[2]        
      
        test_loss = F.nll_loss(F.log_softmax(s_output, dim = 1), y_test).item()
        pred = s_output.data.max(1)[1]      # get the index of the max log-probability   
        correct = pred.eq(y_test.data.view_as(pred)).cpu().sum().item()
        
        y_pred = pred.cpu().numpy()
        y_true = y_test.detach().cpu().numpy()
        
        #'macro'：用于多分类，只有两个属性可以选择 ‘macro’ 和 ‘weighted’ 。
        #' macro '：计算每个标签的指标，并计算它们的未加权平均值。不考虑样本类别是否平衡。
        #' weighted '：计算每个标签的指标，并找到它们的平均值，对(每个标签的真实实例的数量)进行加权。
        recall = recall_score(y_true, y_pred, average='macro') #未加权平均召回率
        w_recall = recall_score(y_true, y_pred, average='weighted')#加权平均召回率
        
  
    print('test loss: {:.2f}, Accuracy: {}/{} ({:.2f}%)\t'.format(
            test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
  
    return correct, recall, w_recall
            
if __name__ == '__main__':
    model = models.my_net(emotion_classes = class_emotion, gender_classes = class_gender, hidden_size=hidden_size)
    correct = 0
    recall = 0
    w_recall = 0
    print(model)
    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct, t_recall, t_w_recall= test(model)
        if t_correct > correct:
            correct = t_correct
        if t_recall > recall:
            recall = t_recall
        if t_w_recall > w_recall:
            w_recall = t_w_recall
            #torch.save(model, 'model.pkl')
        #print('max accuracy{: .2f}%'.format(100. * correct / len_target_dataset ))    
        print('max recall(UAR):{: .2f}%'.format(100. * recall)) 
        print('max w_recall(WAR):{: .2f}%\n'.format(100. * w_recall))





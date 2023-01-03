# -*- coding: utf-8 -*-
"""
Created on Fri May 14 20:07:59 2021

@author: ZHUANG
"""
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_dim=1582, hidden_size=256, out_dim=1582):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=1200),  #1200
            nn.BatchNorm1d(1200,affine=True,track_running_stats=True), 
            nn.ELU(),
            nn.Dropout(),

            nn.Linear(in_features=1200, out_features=900),    #900
            nn.BatchNorm1d(900,affine=True,track_running_stats=True),
            nn.ELU(),        
            nn.Dropout(), 

            nn.Linear(in_features=900, out_features=hidden_size),   #hidden_size=500
            nn.BatchNorm1d(hidden_size,affine=True,track_running_stats=True),
            #nn.ReLU(), 
            nn.ELU(),                                                                                  
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=900),
            nn.BatchNorm1d(900,affine=True,track_running_stats=True),            
            #nn.ELU(),
            nn.Sigmoid(),
            nn.Dropout(),
                      
            nn.Linear(in_features=900, out_features=1200),
            nn.BatchNorm1d(1200,affine=True,track_running_stats=True),
            #nn.ELU(), 
            nn.Sigmoid(),
            nn.Dropout(),
                                           
            nn.Linear(in_features=1200, out_features=out_dim),
            nn.BatchNorm1d(out_dim,affine=True,track_running_stats=True),
            nn.Sigmoid(),  
            #nn.ELU(),           
        )
    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out 

class my_net(nn.Module):    
    def __init__(self, emotion_classes,gender_classes,hidden_size):
        super(my_net, self).__init__()
        self.feature_layers = AutoEncoder()
        self.cls_emotion = nn.Sequential(
            nn.Linear(256, emotion_classes), 
            )
        self.cls_gender = nn.Sequential(
            nn.Linear(256, gender_classes), 
            )
    def forward(self, source, target):
        source_en_out, source_de_out = self.feature_layers(source)
        s_e_p = self.cls_emotion(source_en_out)
        s_g_p = self.cls_gender(source_en_out)
          
        if self.training ==True:
            target_en_out, target_de_out = self.feature_layers(target)           
            t_e_p = self.cls_emotion(target_en_out)
            t_g_p = self.cls_gender(target_en_out)
        else:
            target_en_out = 0
            target_de_out = 0
            t_e_p = 0
            t_g_p = 0

        return source_en_out, source_de_out, s_e_p, s_g_p, target_en_out, target_de_out, t_e_p, t_g_p





















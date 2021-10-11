from load_data import load_data
from Transformer_model import Model
from args import Config
#---------------------------------------------------
import pandas as pd
from collections import Counter
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GroupKFold, KFold
import numpy as np
import torch
from    torch import autograd
import os
from tqdm import tqdm

config = Config()
train,targets,vocabs_size = load_data(config)#加载数据
config.n_vocab = vocabs_size + 1#补充词表大小，词表一定要多留出来一个

batch_size = config.batch_size

kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=2021)#5折交叉验证
for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
    print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
    x_train, x_val = train[train_idx], train[test_idx]
    y_train, y_val = targets[train_idx], targets[test_idx]
    
    M_train = len(x_train)
    M_val = len(x_val)
    if M_train % batch_size == 1:#因为模型里面有层标准化，训练中不能出现单条数据，至少为2条
        M_train -= 1
    if M_val % batch_size == 1:
        M_val -= 1
    x_train = torch.from_numpy(x_train).to(torch.long).to(config.device)
    x_val = torch.from_numpy(x_val).to(torch.long).to(config.device)
    y_train = torch.from_numpy(y_train).to(torch.long).to(config.device)
    y_val = torch.from_numpy(y_val).to(torch.long).to(config.device)

    model = Model(config)#调用transformer的编码器
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    loss_func = nn.CrossEntropyLoss()#多分类的任务
    model.train()#模型中有BN和Droupout一定要添加这个说明
    print('开始迭代....')
    #开始迭代
    for step in range(config.num_epochs):
        print('step=',step+1)
        L_val = -batch_size
        with tqdm(np.arange(0,M_train,batch_size), desc='Training...') as tbar:
            for index in tbar:
                L = index
                R = min(M_train,index+batch_size)
                L_val += batch_size
                L_val %= M_val
                R_val = min(M_val,L_val + batch_size)
                #-----------------训练内容------------------
                train_pre = model(x_train[L:R])     # 喂给 model训练数据 x, 输出预测值
                train_loss = loss_func(train_pre, y_train[L:R])
                val_pre = model(x_val[L_val:R_val])#验证集也得分批次，不然数据量太大内存爆炸
                val_loss = loss_func(val_pre, y_val[L_val:R_val])

                #----------- -----计算准确率----------------
                train_acc = np.sum(np.argmax(np.array(train_pre.data.cpu()),axis=1) == np.array(y_train[L:R].data.cpu()))/(R-L)
                val_acc = np.sum(np.argmax(np.array(val_pre.data.cpu()),axis=1) == np.array(y_val[L_val:R_val].data.cpu()))/(R_val-L_val)

                #---------------打印在进度条上--------------
                tbar.set_postfix(train_loss=float(train_loss.data.cpu()),train_acc=train_acc,val_loss=float(val_loss.data.cpu()),val_acc=val_acc)
                tbar.update()  # 默认参数n=1，每update一次，进度+n

                #-----------------反向传播更新---------------
                optimizer.zero_grad()   # 清空上一步的残余更新参数值
                train_loss.backward()         # 以训练集的误差进行反向传播, 计算参数更新值
                optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    del model





import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

torch.manual_seed(10)#固定每次初始化模型的权重


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               #维度变换(1,28,28) --> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)      #维度变换(16,28,28) --> (16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),                               #维度变换(16,14,14) --> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)      #维度变换(32,14,14) --> (32,7,7)
        )
        self.output = nn.Linear(32*7*7,10)#10个类别，输出10个节点

    def forward(self, x):
        out = torch.from_numpy(x).to(torch.float32)#将输入的numpy格式转换成tensor
        out = self.conv1(out)                  #维度变换(Batch,1,28,28) --> (Batch,16,14,14)
        out = self.conv2(out)                #维度变换(Batch,16,14,14) --> (Batch,32,7,7)
        out = out.view(out.size(0),-1)       #维度变换(Batch,32,14,14) --> (Batch,32*14*14)||将其展平
        out = self.output(out)
        return out


model = CNN()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_func = nn.CrossEntropyLoss()



train_data = np.load('train_data.npz',allow_pickle=True)#加载数据
test_data = np.load('test_data.npz',allow_pickle=True)

x_train = train_data['arr_0'][:,:-1]#训练集
y_train = train_data['arr_0'][:,-1]#训练集标签

x_val = test_data['arr_0'][:,:-1]
y_val = test_data['arr_0'][:,-1]


x_train = x_train.reshape(x_train.shape[0],1,28,28)#转换数据shape
x_val = x_val.reshape(x_val.shape[0],1,28,28)#转换数据shape

training_step = 500#迭代此时
batch_size = 256#每个批次的大小

#开始迭代
for step in range(training_step):
    print('step=',step)
    M_train = len(x_train)
    M_val = len(x_val)
    with tqdm(np.arange(0,M_train,batch_size), desc='Training...') as tbar:
        for index in tbar:
            L = index
            R = min(M_train,index+batch_size)
            #-----------------训练内容------------------
            train_pre = model(x_train[L:R,:])     # 喂给 model训练数据 x, 输出预测值
            train_loss = loss_func(train_pre, torch.from_numpy(y_train[L:R]).to(torch.long))
            val_pre = model(x_val)
            val_loss = loss_func(val_pre, torch.from_numpy(y_val).to(torch.long))
            #----------- -----计算准确率----------------
            train_acc = np.sum(np.argmax(np.array(train_pre.data),axis=1) == y_train[L:R])/(R-L)
            val_acc = np.sum(np.argmax(np.array(val_pre.data),axis=1) == y_val)/M_val

            #---------------打印在进度条上--------------
            tbar.set_postfix(train_loss=float(train_loss.data),train_acc=train_acc,val_loss=float(val_loss.data),val_acc=val_acc)
            tbar.update()  # 默认参数n=1，每update一次，进度+n

            #-----------------反向传播更新---------------
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            train_loss.backward()         # 以训练集的误差进行反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

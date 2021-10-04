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
import cv2
import os
from tqdm import tqdm

torch.manual_seed(10)#固定每次初始化模型的权重

train = pd.read_csv('/零基础入门数据挖掘-心跳信号分类预测/train.csv')
testA = pd.read_csv('/零基础入门数据挖掘-心跳信号分类预测/testA.csv')
sample_submit = pd.read_csv('/零基础入门数据挖掘-心跳信号分类预测/sample_submit.csv')

train['heartbeat_signals'] = train['heartbeat_signals'].apply(lambda x : np.array(x.split(',')).astype('float32'))
train['label'] = train['label'].apply(lambda x : np.array(x).astype('int32'))

testA['heartbeat_signals'] = testA['heartbeat_signals'].apply(lambda x : np.array(x.split(',')).astype('float32'))

data = []
for val in train['heartbeat_signals'].values:
    data.append(val)
data = np.array(data)
targets = train['label'].values

data = data.reshape(data.shape[0],1,205)
#data = torch.from_numpy(data).to(torch.float32)#转换成tensor


test = []
for val in testA['heartbeat_signals'].values:
    test.append(val)
test = np.array(test)
test = test.reshape(test.shape[0],1,205)
test = torch.from_numpy(test).to(torch.float32)#转换成tensor

test_ids = testA['id']
test_pre = np.zeros([len(test),4])
# 对训练集进行切割，然后进行训练
#x_train,x_val,y_train,y_val = train_test_split(data,target,test_size=0.2)

'''
#-----------LSTM主要参数-------
 input_size – 输入的特征维度
 hidden_size – 隐状态的特征维度，也就是LSTM输出的节点数目
 num_layers – 层数（和时序展开要区分开）
 bias – 如果为False，那么LSTM将不会使用偏置，默认为True。
 batch_first – 如果为True，那么输入和输出Tensor的形状为(batch, seq_len, input_size)
 dropout – 如果非零的话，将会在RNN的输出上加个dropout，最后一层除外。
 bidirectional – 如果为True，将会变成一个双向RNN，默认为False。
'''
class Bi_Lstm(nn.Module):
    def __init__(self):
        super(Bi_Lstm,self).__init__() 
        self.lstm = nn.LSTM(input_size = 205, hidden_size = 200,num_layers = 3)#加了双向，输出的节点数翻2倍
        self.l1 = nn.Linear(200,500)#特征输入
        self.l2 = nn.ReLU()#激活函数
        self.l3 = nn.BatchNorm1d(500)#批标准化
        self.l4 = nn.Linear(500,250)
        self.l5 = nn.ReLU()
        self.l6 = nn.BatchNorm1d(250)
        self.l7 = nn.Linear(250,4)#输出4个节点
        self.l8 = nn.BatchNorm1d(4)
    def forward(self, x):
        out,_ = self.lstm(x)
        #选择最后一个时间点的output
        out = self.l1(out[:,-1,:])
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        out = self.l8(out)
        return out


def SoftMax(x):
    return np.e**x/np.sum(np.e**x)

def Score_function(y_pre,y_true):
    score = 0
    y_pre = np.argmax(y_pre,axis=1)
    for i in range(len(y_pre)):
        if y_pre[i] != int(y_true[i]):
            score += 1.0
    return score
training_step = 5#迭代次数
batch_size = 512#每个批次的大小

kf = KFold(n_splits=5, shuffle=True, random_state=2021)#5折交叉验证
for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
    print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
    #print(train_idx)
    x_train, x_val = data[train_idx], data[test_idx]
    y_train, y_val = targets[train_idx], targets[test_idx]

    model = Bi_Lstm()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_func = nn.CrossEntropyLoss()#多分类的任务

    model.train()#模型中有BN和Droupout一定要添加这个说明
    

    #开始迭代
    for step in range(training_step):
        print('step=',step)
        M_train = len(x_train)
        M_val = len(x_val)
        L_val = -batch_size
        with tqdm(np.arange(0,M_train,batch_size), desc='Training...') as tbar:
            for index in tbar:
                L = index
                R = min(M_train,index+batch_size)
                L_val += batch_size
                L_val %= M_val
                R_val = min(M_val,L_val + batch_size)
                #-----------------训练内容------------------
                train_pre = model(torch.from_numpy(x_train[L:R,:]).to(torch.float32))     # 喂给 model训练数据 x, 输出预测值
                train_loss = loss_func(train_pre, torch.from_numpy(y_train[L:R]).to(torch.long))
                val_pre = model(torch.from_numpy(x_val[L_val:R_val,:]).to(torch.float32))#验证集也得分批次，不然数据量太大内存爆炸
                val_loss = loss_func(val_pre, torch.from_numpy(y_val[L_val:R_val]).to(torch.long))
                #----------- -----计算准确率----------------
                train_acc = np.sum(np.argmax(np.array(train_pre.data),axis=1) == y_train[L:R])/(R-L)
                val_acc = np.sum(np.argmax(np.array(val_pre.data),axis=1) == y_val[L_val:R_val])/(R_val-L_val)

                #---------------打印在进度条上--------------
                tbar.set_postfix(train_loss=float(train_loss.data),train_acc=train_acc,val_loss=float(val_loss.data),val_acc=val_acc)
                tbar.update()  # 默认参数n=1，每update一次，进度+n

                #-----------------反向传播更新---------------
                optimizer.zero_grad()   # 清空上一步的残余更新参数值
                train_loss.backward()         # 以训练集的误差进行反向传播, 计算参数更新值
                optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            val_pre = np.array(model(torch.from_numpy(x_val).to(torch.float32)).data)
            y_pre = []
            for val in val_pre:
                y_pre.append(SoftMax(val))
            y_pre = np.array(y_pre)
            print('val_score=',Score_function(y_pre,y_val))
    pre = np.array(model(test).data)
    soft_pre = []
    for val in pre:
        soft_pre.append(SoftMax(val))
    test_pre += np.array(soft_pre)

    del model#删除原来的模型

test_pre /= 5
fp = open('天池submission.csv','w',encoding='utf-8')
fp.writelines('id,label_0,label_1,label_2,label_3\n')
for i in range(len(test_ids)):
    fp.writelines(str(int(test_ids[i])))
    for j in range(4):
        fp.writelines(','+str(test_pre[i][j]))
    fp.writelines('\n')
fp.close()

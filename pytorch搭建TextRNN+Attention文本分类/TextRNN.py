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
from gensim.models.word2vec import Word2Vec

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
#--------------------------加载数据----------------------------
df = pd.read_csv('/新闻文本分类/train_set.csv',sep='\t')


#mx_length = 900
vocabs_size = 0
n_class = 14
training_step = 20#迭代次数
batch_size = 256#每个批次的大小
train = []
targets = []
label = df['label'].values
text = df['text'].values
id = 0
for val in tqdm(text):
    s = val.split(' ')
    single_data = []
    for i in range(len(s)):
        vocabs_size = max(vocabs_size,int(s[i])+1)
        single_data.append(int(s[i])+1)
        if len(single_data)>=256:
            train.append(single_data)
            targets.append(int(label[id]))
            single_data = []
    if len(single_data)>=150:
        single_data = single_data + [0]*(256-len(single_data))
        train.append(single_data)
        targets.append(int(label[id]))  
    id += 1
    


train = np.array(train)
targets = np.array(targets)


class Bi_Lstm(nn.Module):
    def __init__(self):
        super(Bi_Lstm,self).__init__() 
        self.embeding = nn.Embedding(vocabs_size+1,100)
        self.lstm = nn.LSTM(input_size = 100, hidden_size = 100,num_layers = 1,bidirectional = False,batch_first=True,dropout=0.5)#加了双向，输出的节点数翻2倍
        self.l1 = nn.BatchNorm1d(100)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(100,n_class)#特征输入
        self.l4 = nn.Dropout(0.3)
        self.l5 = nn.BatchNorm1d(n_class)
    def forward(self, x):
        x = self.embeding(x)
        out,_ = self.lstm(x)
        #选择最后一个时间点的output
        out = self.l1(out[:,-1,:])
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        return out


print(train.shape)
print(targets.shape)

kf = KFold(n_splits=5, shuffle=True, random_state=2021)#5折交叉验证
for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
    print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
    x_train, x_val = train[train_idx], train[test_idx]
    y_train, y_val = targets[train_idx], targets[test_idx]
    
    M_train = len(x_train)-1
    M_val = len(x_val)

    x_train = torch.from_numpy(x_train).to(torch.long).to(device)
    x_val = torch.from_numpy(x_val).to(torch.long).to(device)
    y_train = torch.from_numpy(y_train).to(torch.long).to(device)
    y_val = torch.from_numpy(y_val).to(torch.long).to(device)

    model = Bi_Lstm()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_func = nn.CrossEntropyLoss()#多分类的任务
    model.train()#模型中有BN和Droupout一定要添加这个说明
    
    #开始迭代
    for step in range(training_step):
        print('step=',step)
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


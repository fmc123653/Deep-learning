import cv2
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import Counter
torch.manual_seed(10)#固定每次初始化模型的权重

#-----------------加载图像数据------------
img = cv2.imread('olivettifaces.jpg')
h = int(img.shape[0]/20)
w = int(img.shape[1]/20)
IMG = []
label = []
id = 0
for i in range(0,20*h,h):
    for j in range(0,20*w,w):
        IMG.append(img[i:i+h,j:j+w,:].reshape(3,h,w)/255)
        label.append(int(id/10))
        id += 1


# 对训练集进行切割，然后进行训练
X_train,X_val,Y_train,Y_val = train_test_split(IMG,label,test_size=0.2)

#-------------生成数据集-----------------

x_train = []
y_train = []
x_val = []
y_val = []
for i in range(len(X_train)):
    for j in range(i+1,len(X_train)):
        if Y_train[i] == Y_train[j]:
            x_train.append([X_train[i],X_train[j]])
            y_train.append(1)
        else:
            key = random.randint(1,10)
            if key>=2:
                continue
            x_train.append([X_train[i],X_train[j]])
            y_train.append(0)

for i in range(len(X_val)):
    for j in range(i+1,len(X_val)):
        if Y_val[i] == Y_val[j]:
            x_val.append([X_val[i],X_val[j]])
            y_val.append(1)
        else:
            key = random.randint(1,10)
            if key>=2:
                continue
            x_val.append([X_val[i],X_val[j]])
            y_val.append(0)

x_train = torch.from_numpy(np.array(x_train)).to(torch.float32)
y_train = np.array(y_train)
x_val = torch.from_numpy(np.array(x_val)).to(torch.float32)
y_val = np.array(y_val)

print('train',Counter(y_train),'val',Counter(y_val))
#------------------搭建网络框架------------
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,#通道数目，刚输入的图片是彩色的三通道数目
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.l1 = nn.Linear(4928,300)#输出300个节点
        self.l2 = nn.Linear(300,1)#输出1个节点
        self.l3 = nn.Sigmoid()
    def forward(self, x1,x2):
        out1 = self.conv1(x1)
        out1 = self.conv2(out1)
        out1 = out1.view(out1.size(0),-1)
        out1 = self.l1(out1)


        out2 = self.conv1(x2)
        out2 = self.conv2(out2)
        out2 = out2.view(out2.size(0),-1)
        out2 = self.l1(out2)

        out = torch.abs(out1-out2)#计算均值误差
        out = self.l2(out)
        out = self.l3(out)
        return out


training_step = 500#迭代次数
batch_size = 256#每个批次的大小
learning_rate = 0.01
model = Siamese()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)#定义优化器
loss_func = nn.BCELoss() #定义损失函数


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
            train_pre = model(x_train[L:R,0],x_train[L:R,1])     # 喂给 model训练数据 x, 输出预测值
            train_loss = loss_func(train_pre, torch.from_numpy(y_train[L:R].reshape(R-L,1)).to(torch.float))
            val_pre = model(x_val[:,0],x_val[:,1])
            val_loss = loss_func(val_pre, torch.from_numpy(y_val.reshape(M_val,1)).to(torch.float))
            #----------- -----计算准确率----------------
            train_acc = np.sum((np.array(train_pre.data)>=0.5)==(y_train[L:R].reshape(R-L,1)>=0.5))/(R-L) 
            val_acc = np.sum((np.array(val_pre.data)>=0.5)==(y_val.reshape(M_val,1)>=0.5))/M_val 

            #---------------打印在进度条上--------------
            tbar.set_postfix(train_loss=float(train_loss.data),train_acc=train_acc,val_loss=float(val_loss.data),val_acc=val_acc)
            tbar.update()  # 默认参数n=1，每update一次，进度+n

            #-----------------反向传播更新---------------
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            train_loss.backward()         # 以训练集的误差进行反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上




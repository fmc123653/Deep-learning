import numpy as np
import pandas as pd
import random
import queue
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
class Node:#每个节点表示一个神经元，每个神经元包括对应的参数
    w=[]#权重参数
    dw=[]#每个参数下降的梯度
    dwc=1
    b=0#常数项
    a=0#输出
    id=0#每个节点的ID
    pos=0#该神经元在所在层的第几个位置

layers = [8,4,2,1]#定义4层的神经网络，第一层为输入层，8个特征，第2层4个神经元，第3层2个神经元，第4层为输出层，1个神经元
lr=0.05
step=1000


node = []
to = [[]for i in range(np.sum(layers[1:]))]
fr = [[]for i in range(np.sum(layers[1:]))]
id=0
for i in range(1,len(layers)):
    for j in range(layers[i]):
        t = Node()
        t.w = np.random.random(layers[i-1])#和上一层的数目对应
        t.dw = np.zeros(layers[i-1])#初始化为0
        t.dwc=1
        t.b = random.random()
        t.a = 0#输出
        t.id = id
        t.pos = j
        id+=1
        node.append(t)#储存下每一个节点

id1 = 0
id2 = 0
for i in range(1,len(layers)-1):#连接节点关系
    id2 += layers[i]
    for j in range(layers[i]):
        for k in range(layers[i+1]):
            to[id1+j].append(id2 + k)
            fr[id2+k].append(id1 + j)
    id1 += layers[i]


def Sigmoid(z):#激活函数
    return 1/(1+np.e**(-z))

def Loss(a,y):#损失函数
    return  -(y*np.log(a)+(1-y)*np.log(1-a))

def forward(x,y,node,layers):#前向传播
    for i in range(layers[1]):
        node[i].a = np.sum(np.array(node[i].w)*x)
    id=0
    for i in range(1,len(layers)):
        for j in range(layers[i]):
            pos = id + j
            node[pos].a = Sigmoid(node[pos].a + node[pos].b)
            for k in range(len(to[id])):
                v = to[id][k]
                node[v].a += node[v].w[node[pos].pos]*node[pos].a
        id += layers[i]
    return Loss(node[np.sum(layers[1:])-1].a,y),node

def backward(x,y,node,layers):#反向传播
    pos=len(node)-1
    node[pos].dwc = (node[pos].a-y)/(node[pos].a*(1-node[pos].a))
    for i in range(len(node[pos].dw)):
        node[pos].dw[i] += node[pos].dwc*node[pos].a*(1-node[pos].a)*node[fr[pos][i]].a
    #利用BFS广度遍历完成反向传播
    q=queue.Queue()
    for i in range(len(node[pos].dw)):
        t=Node()
        t.dwc=node[pos].dwc*node[pos].a*(1-node[pos].a)*node[pos].w[i]
        t.id=fr[pos][i]
        q.put(t)

    while q.empty()==False:
        t=q.get()
        pos=t.id
        for i in range(len(node[pos].dw)):
            if len(fr[pos])==0:#已经反向遍历到第一层
                node[pos].dw[i] += t.dwc*node[pos].a*(1-node[pos].a)*x[i]
            else:
                node[pos].dw[i] += t.dwc*node[pos].a*(1-node[pos].a)*node[fr[pos][i]].a
                t2=Node()
                t2.dwc=t.dwc*node[pos].a*(1-node[pos].a)*node[pos].w[i]
                t2.id=fr[pos][i]
                q.put(t2)
    return node

def Predict_function(x,node,layers):#预测函数
    def Single_predict(x,node,layers):#单条预测
        for i in range(layers[1]):
            node[i].a = np.sum(np.array(node[i].w)*x)
        id=0
        for i in range(1,len(layers)):
            for j in range(layers[i]):
                pos = id + j
                node[pos].a = Sigmoid(node[pos].a + node[pos].b)
                for k in range(len(to[id])):
                    v = to[id][k]
                    node[v].a += node[v].w[node[pos].pos]*node[pos].a
            id += layers[i]
        if node[np.sum(layers[1:])-1].a>=0.5:
            return 1
        else:
            return 0
    y_pred=[]
    for i in range(len(x)):
        y_pred.append(Single_predict(x,node,layers))
    return np.array(y_pred)

N=layers[0]
data=pd.read_csv('diabetes.txt',header=None).values
M=len(data)#数据集大小
x=data[:,0:N]
x=x/(np.max(x)-np.min(x))
y=data[:,-1]

train_x=x[0:int(M*0.8)]
train_y=y[0:int(M*0.8)]
#按8：2划分训练集和验证集
test_x=x[int(M*0.8):]
test_y=y[int(M*0.8):]

cur_los=[]

for index in range(step):
    loss = 0
    for i in range(len(node)):
        for j in range(len(node[i].dw)):
            node[i].dw[j]=0#清空
    M=len(train_x)
    for i in range(M):
        los,node = forward(train_x[i],train_y[i],node,layers)
        loss += los
        node=backward(train_x[i],train_y[i],node,layers)
    #更新权重
    for i in range(len(node)):
        for j in range(len(node[i].dw)):
            node[i].dw[j]/=M
            node[i].w[j]-=lr*node[i].dw[j]#乘学习率更新权重

    cur_los.append(loss/M)
    print('step=',index,'  loss=',loss/M)


y_pre = Predict_function(test_x,node,layers)
print("accuracy_score: %.4lf" % accuracy_score(y_pre,test_y))

plt.plot(cur_los)
plt.title('loss曲线')
plt.show()

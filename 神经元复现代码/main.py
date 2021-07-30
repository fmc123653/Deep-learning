import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
N=8#特征数目
lr=0.00001#学习率
step=100000#迭代次数
w = np.random.rand(1,N+1)[0]*0.01#随机初始化参数,w[N]看成是b

def function(w,x):
    return np.sum(w[0:N]*x)+w[N]

def Sigmoid(z):#激活函数
    return 1/(1+np.e**(-z))

def Loss(w,x,y):#损失函数
    loss=0
    for i in range(len(x)):
        a=Sigmoid(function(w,x[i]))
        loss+=(-(y[i]*np.log(a)+(1-y[i])*np.log(1-a)))
    return loss/len(x)#取平均

def derivative_function(w,x,y):#导数
    derivative=[]
    for j in range(N):
        averge=0
        for i in range(len(x)):
            averge+=Sigmoid(function(w,x[i]))*x[i][j]
        averge/=len(x)#取平均
        derivative.append(averge)#储存下每个参数wj的平均下降梯度
    averge=0
    for i in range(len(x)):#针对参数b
        averge+=Sigmoid(function(w,x[i]))
    averge/=len(x)
    derivative.append(averge)
    return np.array(derivative)
def Predict_function(w,x):
    pred=[]
    for i in range(len(x)):
        res=Sigmoid(function(w,x[i]))
        if res>=0.5:
            pred.append(1)
        else:
            pred.append(0)
    return np.array(pred)
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

los=[]
best_w=w.copy()
best_loss=1e9
for i in range(step):
    loss=Loss(w,train_x,train_y)
    print('step=',i,'   loss=',loss)
    if loss<best_loss:#找最优参数
        best_loss=loss
        best_w=w.copy()
    los.append(loss)
    f=derivative_function(w,x,y)
    w=w-lr*f#乘上学习率，开始梯度下降法

y_pre = Predict_function(best_w,test_x)
print("accuracy_score: %.4lf" % accuracy_score(y_pre,test_y))

plt.plot(los)
plt.show()


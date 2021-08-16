from unicodedata import normalize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras
from keras.layers import Dense,MaxPool2D,Conv2D,Flatten,Dropout,BatchNormalization,Activation,Add,\
                            Input,ZeroPadding2D,MaxPooling2D,AveragePooling2D
from keras.initializers import glorot_uniform
from keras.models import Sequential,Model
from keras.optimizers import Adam

import os
from tqdm import tqdm
from input_data import load_cifar10#导入加载cifat数据的函数
import cv2


X_data = []
Y_data = []

input_shape = (150,150,3)#输入数据格式
num_class = 43#数据类别数目



path='Dataset/GTSRB'
for file in tqdm(os.listdir(path)):
    lab=int(file)
    for photo_file in os.listdir(path+'/'+file):
        if photo_file[0]=='G':
            continue
        photo_file_path=path+'/'+file+'/'+photo_file
        img = cv2.imread(photo_file_path,1)
        img = cv2.resize(img,(input_shape[0],input_shape[1]))
        X_data.append(img)
        Y_data.append(lab)


print(len(X_data))
print(Counter(Y_data))


X_data=np.array(X_data)
X_data=X_data/255.0
Y_data=np.array(Y_data)






# 对训练集进行切割，然后进行训练
train_x,test_x,train_y,test_y = train_test_split(X_data,Y_data,test_size=0.2)

lb=preprocessing.LabelBinarizer().fit(np.array(range(num_class)))#对标签进行ont_hot编码
train_y=lb.transform(train_y)#因为是多分类任务，必须进行编码处理
test_y=lb.transform(test_y)



model = Sequential([
    #第一层
    Conv2D(96,(11,11),padding='valid',strides=(4,4),activation='relu',input_shape=input_shape),#输入图像尺寸为32x32
    BatchNormalization(),#加入批标准化优化模型
    #后面直接跟着池化层
    MaxPooling2D(pool_size=(3,3),strides=(2,2)),
    Dropout(0.5),#加入dropout防止过拟合

    #第二层
    Conv2D(256,(5,5),padding='same',strides=(1,1),activation='relu'),
    BatchNormalization(),
    #后面直接跟着池化层
    MaxPooling2D(pool_size=(3,3),strides=(2,2)),
    Dropout(0.5),#加入dropout防止过拟合

    #第三层
    Conv2D(384,(3,3),padding='same',strides=(1,1),activation='relu'),
    BatchNormalization(),
    Dropout(0.5),#加入dropout防止过拟合

    #第四层
    Conv2D(384,(3,3),padding='same',strides=(1,1),activation='relu'),
    BatchNormalization(),
    Dropout(0.5),#加入dropout防止过拟合

    #第五层
    Conv2D(256,(3,3),padding='same',strides=(1,1),activation='relu'),
    BatchNormalization(),
    #后面直接跟着池化层
    MaxPooling2D(pool_size=(3,3),strides=(2,2)),
    Dropout(0.5),#加入dropout防止过拟合

    #展开所有二维数据矩阵为一维数据核全连接层对接
    Flatten(),

    #第六层为全连接层，4096个节点
    Dense(4096,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),#加入dropout防止过拟合

    #第七层为全连接层，4096个节点；
    Dense(4096,activation='relu'),
    BatchNormalization(),
    Dropout(0.5),#加入dropout防止过拟合

    #第八层为输出层，激活函数为softmax。
    Dense(num_class,activation='softmax')
])

model.summary()#显示模型结构


#编译模型,定义损失函数loss，采用的优化器optimizer为Adam
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#开始训练模型
history=model.fit(train_x,train_y,batch_size = 128,epochs=50,validation_data=(test_x, test_y))#训练1000个批次，每个批次数据量为126


#绘制训练过程的acc和loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend(['acc', 'val_acc'], loc='upper left')
plt.title('AlexNet Acc')
plt.xlabel("Epoch")#横坐标名
plt.ylabel("Accuracy")#纵坐标名

plt.show()
#图像保存方法
plt.savefig('AlexNet acc.png')

plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.legend(['loss', 'val_loss'], loc='upper left')
plt.title('AlexNet Loss')
plt.xlabel("Epoch")#横坐标名
plt.ylabel("Loss")#纵坐标名

plt.show()
#图像保存方法
plt.savefig('AlexNet loss.png')



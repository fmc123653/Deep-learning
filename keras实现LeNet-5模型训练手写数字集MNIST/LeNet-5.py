from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data = np.load('Dataset/mnist.npz')#加载数据

num_class=10#类别数目
input_shape=(28,28,1)#输入的数据尺寸

train_x=[]
train_y=[]
test_x=[]
test_y=[]

print('loading train_data....')
id=0
for val in tqdm(data['x_train']):
    train_x.append(val.reshape(input_shape))
    train_y.append(data['y_train'][id])
    id+=1

print('loading test_data....')
id=0
for val in tqdm(data['x_test']):
    test_x.append(val.reshape(input_shape))
    test_y.append(data['y_test'][id])
    id+=1

train_x=np.array(train_x)/255.0#标准化
train_y=np.array(train_y)
test_x=np.array(test_x)/255.0#标准化
test_y=np.array(test_y)


lb=preprocessing.LabelBinarizer().fit(np.array(range(num_class)))#对标签进行ont_hot编码
train_y=lb.transform(train_y)#因为是多分类任务，必须进行编码处理
test_y=lb.transform(test_y)




model = Sequential([
    #第一层为6个5X5卷积核，步长为1*1，不扩展边界padding为valid，并输入单通道的灰度图像
    Conv2D(6,(5,5),padding='valid',strides=(1,1),activation='tanh',input_shape=input_shape),#输入图像尺寸为32x32
    BatchNormalization(),#加入批标准化优化模型
    #第二层为2X2的最大值池化层，步长为2X2
    MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    #第三层为16个5X5卷积核，步长为1*1，不扩展边界padding为valid
    Conv2D(16,(5,5),padding='valid',strides=(1,1),activation='tanh'),
    BatchNormalization(),
    #第四层为2X2的最大值池化层，步长为2X2
    MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    #第五层为展平层，把前面输出的二维数据矩阵打开展开成一维数据，和全连接层对接
    Flatten(),
    #第六层为全连接层，120个节点
    Dense(120,activation='tanh'),
    BatchNormalization(),
    #第七层为全连接层，84个节点；
    Dense(84,activation='tanh'),
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
plt.title('LeNet-5 Acc')
plt.xlabel("Epoch")#横坐标名
plt.ylabel("Accuracy")#纵坐标名

plt.show()
#图像保存方法
plt.savefig('LeNet-5 acc.png')

plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.legend(['loss', 'val_loss'], loc='upper left')
plt.title('LeNet-5 Loss')
plt.xlabel("Epoch")#横坐标名
plt.ylabel("Loss")#纵坐标名

plt.show()
#图像保存方法
plt.savefig('LeNet-5 loss.png')

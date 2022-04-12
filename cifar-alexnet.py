#导入模块
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,Activation,BatchNormalization,MaxPool2D,Dropout,Dense,Flatten
from tensorflow.keras import datasets
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

#导入数据
cafir10=datasets.cifar10
(x_train,y_train),(x_test,y_test)=cafir10.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

#创建AlexNet网络
class AlexNet(Model):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.c1=Conv2D(filters=96,kernel_size=(3,3))
        self.b1=BatchNormalization()
        self.a1=Activation('relu')
        self.p1=MaxPool2D(pool_size=(3,3),strides=2)

        self.c2=Conv2D(filters=256,kernel_size=(3,3))
        self.b2=BatchNormalization()
        self.a2=Activation('relu')
        self.p2=MaxPool2D(pool_size=(3,3),strides=2)

        self.c3=Conv2D(filters=384,kernel_size=(3,3),padding='same',activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.p3=MaxPool2D(pool_size=(3,3),strides=2)

        self.f1atten=Flatten()
        self.f1=Dense(2040,activation='relu')
        self.d1=Dropout(0.2)
        self.f2 = Dense(2040, activation='relu')
        self.d2=Dropout(0.2)
        self.f3 = Dense(10, activation='relu')

    def call(self,x):
        x=self.c1(x)
        x=self.b1(x)
        x=self.a1(x)
        x=self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x=self.c3(x)

        x=self.c4(x)
        x=self.c5(x)
        x=self.p3(x)

        x=self.f1atten(x)
        x=self.f1(x)
        x=self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y=self.f3(x)
        return y


model=AlexNet()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])

#断点续训
checkpoint_save_path='./checkpoint_AlexNet/AlexNet.ckpt'
if os.path.exists(checkpoint_save_path+'./index'):
    print('----------load model------------')
    model.load_weights(checkpoint_save_path)
cp_back=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,save_weights_only=True,save_best_only=True)

history=model.fit(x_train,y_train,batch_size=32,epochs=4,validation_data=(x_test,y_test),validation_freq=1,callbacks=[cp_back])


file=open('./checkpoint_AlexNet/weights,txt','w')

for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')

file.close()

acc=history.history['sparse_categorical_accurary']
val_acc=history.history['val_sparse_categorical_accurary']
loss=history.history['loss']
val_loss=history.history['val_loss']


plt.subplot(121)
plt.title('training acc and validation acc')
plt.plot(acc,label='Traing acc')
plt.plot(val_acc,label='Validation acc')
plt.legend()

plt.subplot(122)
plt.title('Training and Validation loss')
plt.plot(loss,label='Loss')
plt.plot(val_loss,label='validation Loss')
plt.legend()
plt.show()





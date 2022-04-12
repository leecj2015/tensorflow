#导入模块
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,Activation,BatchNormalization,MaxPool2D,Dropout,Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras import datasets
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

#导入数据
cafir10=datasets.cifar10
(x_train,y_train),(x_test,y_test)=cafir10.load_data()
x_train,x_test=x_train/255.0,x_test/255.0


#创建inceptionNet

class ConvBNRelu(Model):
    def __init__(self,ch,kernelsz=3,strides=1,padding='same'):
        super(ConvBNRelu,self).__init__()
        self.model=tf.keras.models.Sequential([
            Conv2D(ch,kernelsz,strides=strides,padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])
    def call(self,x):
        # 在training=False时，BN通过整个训练集计算均值、方差去做批归一化，
        # training=True时，通过当前batch的均值、方差去做批归一化。
        # 推理时 training=False效果好
        x=self.model(x,training=False)
        return  x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        # concat along axis=channel
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class Inception10(Model):
    def __init__(self,num_blocks,num_classes,init_ch=16,**kwargs):
        super(Inception10,self).__init__(**kwargs)
        self.in_channels=init_ch
        self.out_channel=init_ch
        self.num_blocks=num_blocks
        self.init_ch=init_ch
        self.c1=ConvBNRelu(init_ch)
        self.blocks=tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id==0:
                    block=InceptionBlk(self.out_channel,strides=2)
                else:
                    block=InceptionBlk(self.out_channel,strides=1)
                self.blocks.add(block)
            self.out_channel*=2
        self.p1=GlobalAveragePooling2D()
        self.f1=Dense(num_classes,activation='softmax')
    def call(self,x):
        x=self.c1(x)
        x=self.blocks(x)
        x=self.p1(x)
        y=self.f1(x)
        return  y




model=Inception10(num_blocks=2,num_classes=10)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#断点续训
checkpoint_save_path='./checkpoint_Inception/Inception10.ckpt'
if os.path.exists(checkpoint_save_path+'.index'):
    print('----------load model------------')
    model.load_weights(checkpoint_save_path)
cp_back=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                           save_weights_only=True,
                                           save_best_only=True)

history=model.fit(x_train,y_train,batch_size=32,
                  epochs=4,
                  validation_data=(x_test,y_test),
                  validation_freq=1,
                  callbacks=[cp_back])



file=open('./weights.txt','w')

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





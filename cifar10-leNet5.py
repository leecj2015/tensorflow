import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Dense,Flatten
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#lenet5

class LeNet(Model):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1=Conv2D(filters=6,kernel_size=(5,5),strides=1,activation='sigmoid')
        self.p1=MaxPool2D(pool_size=(2,2),strides=2)

        self.c2=Conv2D(filters=16,kernel_size=(5,5),activation='sigmoid')
        self.p2=MaxPool2D(pool_size=(2,2),strides=2)

        self.flatten=Flatten()
        self.f1=Dense(120,activation='sigmoid')
        self.f2=Dense(84,activation='sigmoid')
        self.f3=Dense(10,activation='softmax')

    def call(self,x):
        x=self.c1(x)
        x=self.p1(x)

        x=self.c2(x)
        x=self.p2(x)

        x=self.flatten(x)
        x=self.f1(x)
        x=self.f2(x)
        y=self.f3(x)
        return y


model=LeNet()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path='./checkpoint-lenet5/LeNet5.ckpt'
if os.path.exists(checkpoint_save_path+'.index'):
    print('--------------load model--------------')
    model.load_weights(checkpoint_save_path)
cp_back=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                           save_best_only=True,
                                           save_weights_only=True)
history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1,callbacks=[cp_back])
model.summary()

file=open('./checkpoint/weights.txt','w')
for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.value)+'\n')
    file.write(str(v.numpy())+'\n')
file.close()
acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.subplot(121)
plt.title('Training and Validation Accuracy')
plt.plot(acc,label='Training acc')
plt.plot(val_acc,label='validation acc')
plt.legend()

plt.subplot(122)
plt.title('Training and Validation loss')
plt.plot(loss,label='Loss')
plt.plot(val_loss,label='validation Loss')
plt.legend()
plt.show()



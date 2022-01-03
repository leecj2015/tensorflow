import numpy as np
import tensorflow as tf
from tensorflow import keras
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
print(train_images.shape,train_labels.shape,test_images.shape,test_labels.shape)
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
#显示图片
import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#归一化
train_images=train_images/255.0
test_images=test_images/255.0
#显示前面二十五张图片，分别5行5列
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
#建立模型
model=tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
#设置参数并进行训练
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=5)

#模型评估
test_Loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
print('\n Test Acc:',test_acc)
print('\n Test_Loss:',test_Loss)
predictions=model.predict(test_images)
print(predictions.shape)
print(predictions[0])
print(np.argmax(predictions[0]))

def plot_Image(i,prediction_array,true_label,img):
    prediction_array,true_label,img=prediction_array,true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap=plt.cm.binary)
    prediction_label=np.argmax(prediction_array)
    if prediction_label==true_label:
        color='blue'
    else:
        color='red'
    plt.xlabel('{} {:2.0f}%({})'.format(class_names[prediction_label],
                                        100*np.max(prediction_array),
                                        class_names[true_label]),
                                        color=color)

def plot_value_array(i,prediction_array,true_label):
    prediction_array,true_label=prediction_array,true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot=plt.bar(range(10),prediction_array,color='#777777')
    plt.ylim([0,1])
    prediction_label=np.argmax(prediction_array)
    thisplot[prediction_label].set_color('red')
    thisplot[true_label].set_color('blue')
# i=0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_Image(i,predictions[i],test_labels,test_images)
# plt.subplot(1,2,2)
# plot_value_array(i,predictions[i],test_labels)
# plt.show()


i=12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_Image(i,predictions[i],test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i,predictions[i],test_labels)
plt.show()

#保存训练好的模型
#保存权重参数与网络模型
model.save('fashion_model.h5')
#网咯架构
config=model.to_json()
print(config)
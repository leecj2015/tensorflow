import tensorflow as tf
from  tensorflow import keras
model=keras.models.load_model('fashion_model.h5')
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
predicitons=model.predict(test_images)
print(predicitons.shape)


import matplotlib.pyplot as plt
import numpy as np
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


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



'''
测试时候进行相同的热处理
'''
train_images=train_images/255.0
test_images=test_images/255.0
predicitons=model.predict(test_images)
num_rows=5
num_clos=3
num_images=num_rows*num_clos
plt.figure(figsize=(4*num_clos,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_clos,2*i+1)
    plot_Image(i,predicitons[i],test_labels,test_images)
    plt.subplot(num_rows,2*num_clos,2*i+2)
    plot_value_array(i,predicitons[i],test_labels)
plt.tight_layout()
plt.show()

config=model.to_json()
with open('config.json','w')as json:
    json.write(config)
model=keras.models.model_from_json(config)
# print(model.summary())

#權重參數
weights=model.get_weights()
print(weights)
model.save_weights('weights.h5')
model.load_weights('weighis.h5')
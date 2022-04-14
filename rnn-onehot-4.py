'''
利用RNN 连续输入四个字母预测下一个字母：
abcd->e
bcde->a
cdea->c
.....

'''

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense,SimpleRNN
from matplotlib import pyplot as plt

input_words='abcde'
word_to_id={'a':0,'b':1,'c':2,'d':3,'e':4}#单词隐射到ID
id_to_onehot={0:[1.,0.,0.,0.,0],1:[0.,1.,0.,0.,0.],2:[0.,0.,1.,0.,0.],3:[0.,0.,0.,1.,0.],4:[0.,0.,0.,0.,1.],}


x_train=[
    [id_to_onehot[word_to_id['a']],id_to_onehot[word_to_id['b']],id_to_onehot[word_to_id['c']],id_to_onehot[word_to_id['d']]] ,
    [id_to_onehot[word_to_id['b']],id_to_onehot[word_to_id['c']],id_to_onehot[word_to_id['d']],id_to_onehot[word_to_id['e']]] ,
    [id_to_onehot[word_to_id['c']],id_to_onehot[word_to_id['d']],id_to_onehot[word_to_id['e']],id_to_onehot[word_to_id['a']]] ,
    [id_to_onehot[word_to_id['d']],id_to_onehot[word_to_id['e']],id_to_onehot[word_to_id['a']],id_to_onehot[word_to_id['b']]] ,
    [id_to_onehot[word_to_id['e']],id_to_onehot[word_to_id['a']],id_to_onehot[word_to_id['b']],id_to_onehot[word_to_id['c']]],
        ]
y_train=[word_to_id['e'],word_to_id['a'],word_to_id['b'],word_to_id['d'],word_to_id['d']]

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
#(sample, steps  input_shape)->(len(x_train),4,5)
x_train=np.reshape(x_train,(len(x_train),4,5))
y_train=np.array(y_train)

model=tf.keras.Sequential([SimpleRNN(3),Dense(5,activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path='./checkpointRnn/rnn_onehot.ckpt'
if os.path.exists(checkpoint_save_path+'.index'):
    print('-----------loade model-----------')
    model.load_weights(checkpoint_save_path)
cp_call=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,save_best_only=True,save_weights_only=True,monitor='loss')

history=model.fit(x_train,y_train,batch_size=32,epochs=50,callbacks=[cp_call])
model.summary()

preNum=int(input('input the number of test alphabet:'))
for i in range(preNum):
    alphabet1=input('input test alpahbet:')
    alphabet=[id_to_onehot[word_to_id[a]] for a in alphabet1]
    alphabet=np.reshape(alphabet,(1,4,5))
    result=model.predict([alphabet])
    pred=tf.argmax(result,axis=1)
    pred=int(pred)
    tf.print(alphabet1+'->'+input_words[pred])


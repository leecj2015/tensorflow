import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

cafir10=tf.keras.datasets.cifar10
(x_train,y_trian),(x_test,y_test)=cafir10.load_data()

#可视化训练集输入特征的第一个元素
plt.imshow(x_train[0])
plt.show()
print('x_trian[0]\n',x_train[0])
print('y_train[0]\n',y_trian[0])

print('x_trian.shape\n',x_train.shape)
print('y_train.shape\n',y_trian.shape)
print('x_test.shape\n',x_test.shape)
print('y_test.shape\n',y_test.shape)
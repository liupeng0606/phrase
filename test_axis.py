import numpy as np
from keras.layers import concatenate
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error
from keras.utils import to_categorical
import pandas as pd
# y_true=[1.0,1.0]
# y_pred=[2.0,3.0]
#
# x = mean_squared_error(y_true,y_pred)
#
# print(x)
# sampled_labels = range(1,400)
# sampled_labels = to_categorical(sampled_labels, num_classes=400)
# print(sampled_labels.shape)
idx = np.random.randint(0, 10, 3)

print(idx)
print(len(idx))

#
#
# x = np.random.rand(3)
# y = np.random.rand(2)
#
#
# input = Input(shape=(3,), dtype='float32')
# pre_y = Dense(2)(input)
#
#
# z = concatenate([input, pre_y], axis=1)
#
#
#
# print(z)
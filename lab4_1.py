import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

train = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab4_1_train.csv')
test = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab4_1_test.csv')

#x_train = train['x_train_'+str(0)].values
x_train = np.array([])
x_test = np.array([])

for i in range(0,5):
  temp = train['x_train_' + str(i)].values
  x_train = np.concatenate([x_train, temp])

y_train = train['y_train'].values

for i in range (0, 5):
  temp_te = test['x_test_' + str(i)].values
  x_test = np.concatenate((x_test, temp_te))

print(x_test)

#data = np.concatenate((train, test), axis = 0)
#print(data)

#print(x_train.shape[0])
#x_train = x_train.reshape(x_train, (x_train[0], 1, x_train.shape[1]))
#model = models.Sequential()
#model.add(layers.LSTM(10, input_shape=(6,)))
#model.add(layers.Dense(1, activation='sigmoid'))
#print(model.summary())

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

train = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab4_3_train.csv')
test = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab4_3_test.csv')
actual = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab4_3.csv')

x_train = train.drop(['y_train'], axis=1)
x_test = test.drop(['y_test'], axis=1)
y_train = train['y_train']
y_test = test['y_test']

x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)
x_train = x_train.values.reshape(x_train.shape[0], x_test.shape[1], 1)

model = models.Sequential()
model.add(layers.LSTM(100, activation='relu', input_shape=(5, 1)))
model.add(layers.Dense(1))
adam = optimizers.Adam(lr=0.001)
model.compile(optimizer='adam',loss='mse')
model.summary()


history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test),verbose=1)

lo = actual['close']
current_closing = actual.drop(['volume'], axis=1)
current_closing['date'] = (current_closing['open'] + current_closing['high'] + current_closing['low'] + current_closing['close']) / 4
print(current_closing)
current_closing = current_closing.values.reshape(current_closing.shape[0], 5, 1)
predicted = model.predict(current_closing, verbose=0)

for i in range(len(predicted)):
  print("Current_closing: ", lo[i])
  print(f"Predicted closing: {predicted[i]}")

print("Test_error: ", model.evaluate(x_test,  y_test, verbose=2))
print("Train_error: ", model.evaluate(x_train,  y_train, verbose=2))

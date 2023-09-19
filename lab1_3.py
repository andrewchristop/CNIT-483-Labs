# Some useful hint. Please feel free to program without the hint
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Import Data
data=pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Lab1_3.csv')
x_data = data[['Open','High','Low']]
y_data = data['Close']
# Generate training data (70% of the given data samples) and the testing data (30% of the given data samples). You can change to other percentage value as long as test_size <=0.3.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

model = models.Sequential()
model.add(layers.Normalization(input_shape=(3,), axis=None))
model.add(layers.Dense(1))
model.summary()

adam = optimizers.Adam(learning_rate=0.5)
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

history = model.fit(x_train, y_train, epochs=3500,
                    validation_data=(x_test, y_test))

W = model.layers[1].get_weights()
print(W)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print("Test_error: ", test_acc)
print("Train_error: ", train_acc)

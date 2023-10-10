import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_HW1_1.csv')
x_data = data['x'].values
y_data = data['y'].values

x_squared = x_data ** 2

joined = np.stack((x_data, x_squared), axis = 1)

x_train, x_test, y_train, y_test = train_test_split(joined, y_data, test_size=0.3)

model = models.Sequential()
model.add(layers.Normalization(input_shape=(2,), axis=None))
model.add(layers.Dense(1))
model.summary()

adam = optimizers.Adam(learning_rate=0.5)
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

history = model.fit(x_train, y_train, epochs=2000,
                    validation_data=(x_test, y_test))

W = model.layers[1].get_weights()
print(W)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print("Test_error: ", test_acc)
print("Train_error: ", train_acc)



import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab2_2.csv')
x_data = data[['x_data_0', 'x_data_1']]
y_data = data['y_data']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.7)


model = models.Sequential()
model.add(layers.Input(shape=(2,))) #Designates input variable to have 2 dimensions
model.add(layers.Dense(10, activation='relu')) #First layer has 10 computing nodes and uses the relu function
model.add(layers.Dense(1, activation='sigmoid'))#Second layer has 1 computing node and uses the sigmoid function
model.summary()

adam = optimizers.Adam(learning_rate=0.3) #Implements the adam optimizer with a learning rate of 0.3
model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy']) #Uses a logarithmic loss function and calculates the model's accuracy 

history = model.fit(x_train, y_train, epochs=300,
                    validation_data=(x_test, y_test)) #Iterates 300 times during the training of the model
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print("Test_error: ", test_loss)
print("Train_error: ", train_loss)



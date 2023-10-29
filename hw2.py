import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras import layers, models, optimizers
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

(x_train, y_train), (x_test, y_test) = mnist.load_data() #Loads the MNIST data set
x_data = np.concatenate((x_train, x_test)) #Temporary concatenation to prepare for train_test_split() 
y_data = np.concatenate((y_train, y_test)) #Temporary concatenation to prepare for train_test_split()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.7) #Training data is 70% of total set

#x_train = x_train / 255.0
#x_test = x_test / 255.0

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28))) #Flatten the input images to 28 by 28 to prepare them for FCNN processing
model.add(layers.Dense(10, activation='relu')) #First layer has 10 computing nodes and uses the relu function
model.add(layers.Dense(20, activation='softmax')) #Second layer has 20 computing node and uses the softmax function
model.summary()
adam = optimizers.Adam(learning_rate=0.3) #Implements the adam optimizer with a learning rate of 0.3

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']) #Uses a logarithmic loss function and calculates the model's accuracy 

history = model.fit(x_train, y_train, epochs=300,
                    validation_data=(x_test, y_test)) #Iterates 300 times during the training of the model
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print("Test_error: ", test_loss)
print("Train_error: ", train_loss)


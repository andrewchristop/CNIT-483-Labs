import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, losses
from sklearn import metrics
from sklearn.model_selection import train_test_split

#CNN function definition
def cnn():
  model = models.Sequential()

  #Alternating convolutional and max pooling layers
  #Convolutional layers extract learn-able features while max-pooling ones downsamples
  model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3))) #1st conv. layer
  model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same')) #1st max-pooling layer
  model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3))) #2nd conv. layer
  model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same')) #2nd max-pooling layer
  model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3))) #3rd conv. layer
  model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same')) #3rd max-pooling layer
  model.add(layers.Flatten())
  model.add(layers.Dense(50, activation='relu')) #ReLU layer has 10 computing nodes
  model.add(layers.Dense(45, activation='softmax')) #Softmax layer has 10 computing nodes
  model.summary()

  model.compile(optimizer='adam',
                loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']) 
  history = model.fit(x_train, y_train, epochs=20,
                      validation_data=(x_test, y_test))
  
  #Evaluation of train-test loss

  test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
  train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
  
  print("Test_error conducted on CNN: ", test_loss)
  print("Train_error conducted on CNN: ", train_loss)

#FCNN function declaration
def fcnn():
  model = models.Sequential()
  model.add(layers.Flatten(input_shape=(32,32,3))) #Flattens data to prepare for FCNN
  model.add(layers.Dense(60, activation='relu')) #1st ReLU layer with 60 nodes 
  model.add(layers.Dense(50, activation='relu')) #2nd ReLU layer with 50 nodes
  model.add(layers.Dense(45, activation='softmax')) #3rd softmax layer with 45 nodes
  model.summary()
  adam = optimizers.Adam(learning_rate=0.3) 
  
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']) 
  
  history = model.fit(x_train, y_train, epochs=100,
                      validation_data=(x_test, y_test))

  #Evaluation of train-test loss
  test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
  train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
  print("Test_error on FCNN: ", test_loss)
  print("Train_error on FCNN: ", train_loss)

#Loading CIFAR-10 dataset into train test pairs
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data();

#Divided by 255 to change pixel intensity value to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#fcnn()
cnn()

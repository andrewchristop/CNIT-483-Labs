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
  model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))) #1st conv. layer
  model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same')) #1st max-pooling layer
  model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))) #2nd conv. layer
  model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same')) #2nd max-pooling layer
  model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))) #3rd conv. layer
  model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same')) #3rd max-pooling layer
  model.add(layers.Flatten()) #Flattening layers to prepare for FCNN
  model.add(layers.Dense(10, activation='relu')) #ReLU layer has 10 computing nodes 
  model.add(layers.Dense(10, activation='softmax'))#Softmax layer has 10 computing nodes
  model.summary()

  model.compile(optimizer='adam',
                loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']) 
  history = model.fit(x_train, y_train, epochs=10,
                      validation_data=(x_test, y_test))
  
  #Evaluation of train-test loss
  test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
  train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
  
  print("Test_error conducted on CNN: ", test_loss)
  print("Train_error conducted on CNN: ", train_loss)

#FCNN function definition
def fcnn():
  model = models.Sequential()
  model.add(layers.Flatten(input_shape=(28,28,1)))#Flatten the input images to 28 by 28 to prepare them for FCNN processing 
  model.add(layers.Dense(10, activation='relu')) #First layer has 10 computing nodes and uses the relu function
  model.add(layers.Dense(10, activation='softmax')) #Second layer has 10 computing nodes and uses the relu function
  model.summary()
  adam = optimizers.Adam(learning_rate=0.3) 
  
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']) 
  
  history = model.fit(x_train, y_train, epochs=10,
                      validation_data=(x_test, y_test))
  
  #Evaluation of train-test loss

  test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
  train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
  print("Test_error on FCNN: ", test_loss)
  print("Train_error on FCNN: ", train_loss)

#Loading MNIST fashion dataset into train test pairs
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data();

#Reshaping the data
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0 
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0 

#FCNN and CNN function calls
fcnn()
cnn()

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, losses
from sklearn import metrics
from sklearn.model_selection import train_test_split

def cnn():
  model = models.Sequential()
  model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same'))
  model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same'))
  model.add(layers.Flatten())
  model.add(layers.Dense(10, activation='relu')) 
  model.add(layers.Dense(20, activation='relu'))
  model.summary()

  model.compile(optimizer='adam',
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']) 
  history = model.fit(x_train, y_train, epochs=20,
                      validation_data=(x_test, y_test))
  
  test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
  train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
  
  print("Test_error conducted on CNN: ", test_loss)
  print("Train_error conducted on CNN: ", train_loss)

def fcnn():
  model = models.Sequential()
  model.add(layers.Flatten(input_shape=(32,32,3))) 
  model.add(layers.Dense(10, activation='relu')) 
  model.add(layers.Dense(20, activation='softmax')) 
  model.summary()
  adam = optimizers.Adam(learning_rate=0.3) 
  
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']) 
  
  history = model.fit(x_train, y_train, epochs=20,
                      validation_data=(x_test, y_test)) 
  test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
  train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
  print("Test_error on FCNN: ", test_loss)
  print("Train_error on FCNN: ", train_loss)

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data();

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

fcnn()
cnn()

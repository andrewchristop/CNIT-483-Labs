import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

train = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab4_3_train.csv') #Import training data and saving it to train numpy array
test = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab4_3_test.csv') #Import testing data and saving it to test numpy array
actual = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab4_3.csv') #Import actual dadta and saving it to actual numpy array

x_train = train.drop(['y_train'], axis=1) #Dropping y_train column from train array to exclusively save x_train data
x_test = test.drop(['y_test'], axis=1) #Dropping y_test column from test array to exclusively save x_test data
y_train = train['y_train'] #Saving y_train column from train array to y_train array
y_test = test['y_test'] #Saving y_test column from test array to y_test array

x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1) #Reshaping data 
x_train = x_train.values.reshape(x_train.shape[0], x_test.shape[1], 1) #Reshaping data

model = models.Sequential()
model.add(layers.LSTM(100, activation='relu', input_shape=(5, 1))) #Importing LSTM layer with 100 computing nodes, using relu activation function
model.add(layers.Dense(1)) #Dense layer
adam = optimizers.Adam(lr=0.001) #Using adam optimizer with a learning rate of 0.001
model.compile(optimizer='adam',loss='mse') #Model compilation using adam optimizer and mean squared error loss function
model.summary() #Prints model summary

history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test),verbose=1) #Running the model at 200 epochs

lo = actual['close'] #Saving close column in a new numpy array
current_closing = actual.drop(['volume'], axis=1) #Dropping volume column from actual array and saving it to current_closing
current_closing['date'] = (current_closing['open'] + current_closing['high'] + current_closing['low'] + current_closing['close']) / 4 #Replacing the date column with average value of 4 other columns
current_closing = current_closing.values.reshape(current_closing.shape[0], 5, 1) #Reshaping actual data to fit training model shape
predicted = model.predict(current_closing, verbose=0) #Predicting values from actual data

for i in range(len(predicted)): #Looping through predicted data array
  print("Current_closing: ", lo[i]) #Prints current iteration of the current closing value
  print(f"Predicted closing: {predicted[i]}") #Prints the predicted closing value of the current iteration

print("Test_error: ", model.evaluate(x_test,  y_test, verbose=2)) #Printing training error
print("Train_error: ", model.evaluate(x_train,  y_train, verbose=2)) #Printing testing error

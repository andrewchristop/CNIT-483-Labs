import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import models, layers

max_features = 20000 #Considering top 20000 words
max_len = 200 #Maximum length of review limited to 200 words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) #Loading train-test data from IMDb dataset

x_train = sequence.pad_sequences(x_train, maxlen=max_len) #Padding sequences to a fixed length for x_train
x_test = sequence.pad_sequences(x_test, maxlen=max_len) #Padding sequences to a fixed length for x_test

model = models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len)) #Embedding layer maps word indices to dense vectors of a fixed size 
model.add(layers.GRU(64)) #Gated Recurrent Unit with 64 computing nodes
model.add(layers.Dense(1, activation='sigmoid')) #Output layer with sigmoid activation function

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #Compiling the model
model.summary() #Printing the model summary

history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test)) #Running the model at 3 epochs

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2) #Model evaluation for test data
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2) #Model evaluation for train data

print("Test_error: ", test_loss) #Printing test error
print("Train_error: ", train_loss) #Printing train error


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models, optimizers, losses
from sklearn import metrics
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data();

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0 
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.0 

model = models.Sequential()
model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(25, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same'))
model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same'))
model.add(layers.MaxPooling2D(1, strides = 1, padding = 'same'))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu')) 
model.add(layers.Dense(20, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']) 
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)

print("Test_error: ", test_loss)
print("Train_error: ", train_loss)



import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data=pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Lab1_2.csv') # you may need to change the path
x_data = data['x'].values
y_data = data['y'].values

joined = np.stack((x_data, y_data), axis=1)
# Generate training data (70% of the given data samples) and the testing data (30% of the given data samples). You can change to other percentage value as long as test_size <=0.3.
x_train, x_test, y_train, y_test = train_test_split(joined, y_data, test_size=0.3)

model = models.Sequential()
model.add(layers.Normalization(input_shape=(2,), axis=None))
model.add(layers.Dense(1))
model.summary()

# Build learning model by using gradient-descent method
adam = optimizers.Adam(learning_rate=0.5)
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

history = model.fit(x_train, y_train, epochs=3500,
                    validation_data=(x_test, y_test))
W = model.layers[1].get_weights()
print(W)

plt.plot(history.history['mean_absolute_error'], label='Train_error')
plt.plot(history.history['val_mean_absolute_error'], label = 'Test_error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.xlim([0, 2000])
plt.ylim([0, 20])
plt.legend(loc='lower right')
plt.show(block=True)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print("Test_error: ", test_acc)
print("Train_error: ", train_acc)

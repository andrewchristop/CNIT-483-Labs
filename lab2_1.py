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

data = pd.read_csv('/Users/christopherandrew/Documents/CNIT 483/CNIT-483-Lab/Data_Lab2_1.csv')
x_data = data[['x_data_0', 'x_data_1']]
y_data = data['y_data']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing import sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x_train = pd.read_csv('/home/cbudihar/CNIT-483-Labs/Data_Lab4_1_train.csv')
x_test = pd.read_csv('/home/cbudihar/CNIT-483-Labs/Data_Lab4_1_test.csv')





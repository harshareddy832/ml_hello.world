import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
cool= Dense(units=1, input_shape=[1])
model = Sequential([cool])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([7.0, 0.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([10.0, -4.0, 0.0, 2.0, 4.0, 6.0], dtype=float)
model.fit(xs, ys, epochs=625)
print(model.predict([11.0]))
print("Learnt: {}",format(cool.get_weights()))

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 13:50:53 2018

@author: Jane Wu
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

N = 200;
X = np.linspace(-1, 1, N)
np.random.shuffle(X)
Y = 3*X + 0.5 +  np.random.normal(0, 0.4, (N,))

X_train, Y_train = X[:60], Y[:60]
X_test, Y_test = X[60:], Y[60:]

#plt.scatter(X, Y)
#plt.show()

model = Sequential()
model.add(Dense(output_dim=1, input_dim = 1))
model.compile(loss='mse', optimizer='sgd')


for k in np.arange(1, 51): 
 Hist = model.fit(X_train, Y_train, batch_size=N, epochs=k, verbose=0)
#print(K.history)
 W, b = model.layers[0].get_weights()
 print('For epoch =', k, 'Weight=', W, ', bias=', b)

 Y_pred = model.predict(X_test)
 plt.scatter(X_test, Y_test)
 plt.plot(X_test, Y_pred)
 plt.show()
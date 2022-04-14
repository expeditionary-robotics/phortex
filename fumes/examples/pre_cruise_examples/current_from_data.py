"""Demonstrates how to define current frunctions from data."""

import numpy as np
import matplotlib.pyplot as plt

from fumes.environment.utils import curfunc, headfunc
from fumes.environment.current import CurrMag, CurrHead

# create data objects
T = np.linspace(0, 12*3600, 1000)
cur = curfunc(None, T) + np.random.normal(0, 0.1, T.shape)
head = headfunc(T) + np.random.normal(0, 0.01, T.shape)

# create current object
currmag = CurrMag(T, cur, training_iter=100, learning_rate=0.1)
currhead = CurrHead(T, head, training_iter=100, learning_rate=0.1)

# get mean and samples
testT = np.linspace(0, 12*3600, 2000)
cur_mean = currmag.magnitude(None, testT)
cur_samps = currmag.sample(testT, num_samples=100)
plt.plot(testT, cur_mean, c='r', label='Mean')
for i in range(100):
    plt.plot(testT, cur_samps[i,:], c='k', alpha=0.1, label='Samples')
plt.plot(testT, curfunc(None, testT), c='g', label="True")
plt.scatter(T, cur, c='g', alpha=0.1, label="Data")
plt.legend(['Mean', 'Samples', 'True', 'Data'])
plt.show()

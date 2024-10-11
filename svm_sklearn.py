import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])

plt.scatter(X, y)
plt.savefig('svm_sklearn.png')

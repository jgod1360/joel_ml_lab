# Developer: Joel GODDOT


import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

# Getting random data
random_state = np.random.RandomState(10)
x = 6 * random_state.rand(30)
y = 5 * x - 0.5 + random_state.rand(30)  # y = 5x - 4

linear_regr = LinearRegression(fit_intercept=False)
X = x[:, np.newaxis]
linear_regr.fit(X, y)
lspace = np.linspace(0, 5)
X_regr = lspace[:, np.newaxis]
y_regr = linear_regr.predict(X_regr)
plt.scatter(x, y);
plt.plot(X_regr, y_regr)
plt.show()

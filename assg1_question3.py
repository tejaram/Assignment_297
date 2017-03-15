import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
from pandas import *

def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

def mean(values):
	return sum(values) / float(len(values))


def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

def variance(values, mean):
	return sum([(x-mean)**2 for x in values])




dataset = [[-1, -1.2], [-1, -2.4], [-1, -3.3], [-1, -4.4], [2.2, -1.1],[2.4,-2.6],
[2.5,3.5],[2.4,-4.7],[3.5,1.4],[3.6,1.5],[3.2,-3.8],[3.4,-4.1],[4.3,6.6],[4.1,2.5],[4.3,-3.3],[2.5,8.5],[5.3,1.5],[5.4,2.8],[5.1,3.2],[5.7,4.9]]

b0, b1 = coefficients(dataset)
print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))

def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions

ymin, ymax = plt.ylim()


xx = np.linspace(ymin, ymax)
yy = b1 * xx + b0
 
# Plot the hyperplane
plt.plot(xx,yy, 'k-')
plt.ylim([0,8]) # Limit the y axis size
plt.ylim([0,6])
plt.show()
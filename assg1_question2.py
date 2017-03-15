import numpy as np
import matplotlib.pyplot as mpl
from sklearn.linear_model import perceptron
from pandas import *

# Create a dataframe using the pandas

data = DataFrame({
'A' :       [-1,-1,-1,-1,2.2,2.4,2.5,2.4,3.5,3.6,3.2,3.4,4.3,4.1,4.3,2.5,5.3,5.4,5.1,5.7],
'B' :       [-1.2,-2.4,-3.3,-4.4,-1.1,-2.6,-3.5,-4.7,1.4,1.5,-3.8,-4.1,6.6,2.5,-3.3,8.5,1.5,2.8,3.2,4.9],
'Targets' : [-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,1,1,1,1]
})

# create a colormap of array

colormap = np.array(['r','b', 'k'])

# Here we are creating a preceptron with iterations and increamental function with 0.004 

net = perceptron.Perceptron(n_iter=60, verbose=0, random_state=None, fit_intercept=True, eta0=0.004)
net.fit(data[['A', 'B']],data['Targets'])
mpl.scatter(data.A, data.B, c=colormap[data.Targets], s=40)

ymin, ymax = mpl.ylim()
print (net.coef_)
w = net.coef_[0]
print (w)
a = -w[0] / w[1]
xx = np.linspace(ymin, ymax)
yy = a * xx - (net.intercept_[0]) / w[1]
 
# plot the graph using the mat plot lib.
mpl.plot(xx,yy, 'k-')
mpl.ylim([0,8]) 
mpl.show()
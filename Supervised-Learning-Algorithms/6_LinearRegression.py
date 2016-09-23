#linear regression

import matplotlib.pyplot as plt
from random import randint
import math
import numpy as np

filename = 'linear.txt' #Read the data set training file
fileP = open(filename, 'rU')

temp = [r.split(',')  for r in fileP.read().split()]
dataPoints = [[]]
dataPoints =  [ [1,float(t[0]),float(t[1])] for t in temp]#type cast each point to float

dataPoints1 =[ [float(t[2])] for t in temp]




X = np.array(dataPoints)
Y=np.array(dataPoints1)
print "Input Matrix"
print X

#print Y


a= (X.T.dot(X))

a = np.asmatrix(a).I
b=X.T.dot(Y)

c=a.dot(b)


print "Final weight vector"
print c



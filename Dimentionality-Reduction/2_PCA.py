#PCA


import matplotlib.pyplot as plt
from random import randint
import math
from numpy import matrix
import numpy as np
import pandas as pd

filename = 'dims.txt' #Read the data set training file
fileP = open(filename, 'rU')

temp = [r.split(',')  for r in fileP.read().split()]
dataPoints = [[]]
dataPoints =  [ [float(t[0]),float(t[1]),float(t[2]),float(t[3])] for t in temp]



#print dataPoints
a = np.array(dataPoints)

print "#######Input Vector##########"
print a
mean_vec = np.mean(a, axis=0)
cov_mat = (a - mean_vec).T.dot((a - mean_vec)) / (a.shape[0]-1)
print '#####Covariance matrix #######\n%s' %cov_mat

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print '##########Eigenvectors####### \n%s' %eig_vecs
print '##########\nEigenvalues########## \n%s' %eig_vals 

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#print '\nEigenpairs \n%s' %eig_pairs 
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
#print eig_pairs
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print '########Eigenvalues in descending order:##########' 
for i in eig_pairs:
    print(i[0])

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print '######Matrix of top 2 eigen vectors#######'
print matrix_w

Y = a.dot(matrix_w)
print "########Final Dimensionally reduced vector"
print Y

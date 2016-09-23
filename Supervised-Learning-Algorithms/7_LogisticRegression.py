#Logistic Regression
from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
from random import randint
import math
import numpy as np
def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))
filename = 'linear.txt' #Read the data set training file
fileP = open(filename, 'rU')

temp = [r.split(',')  for r in fileP.read().split()]
dataPoints = [[]]
dataPoints =  [ [float(t[0]),float(t[1]),float(t[2])] for t in temp]#type cast each point to float

y =[ float(t[4]) for t in temp]

#print dataPoints

X = np.array(dataPoints)
y=np.array(y)
Xpred=X


theta_values = [0,1,1,1]


theta_values=np.array(theta_values)


def logistic_func(theta, x):
    return 1.0 / (1 + np.exp(-x.dot(theta)))


def log_gradient(theta, x, y):

	first_calc = logistic_func(theta, x) - np.squeeze(y)
	
	final_calc = first_calc.T.dot(x)

	return final_calc


def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
  
    y = np.squeeze(y)
    # print y
    step1 = y * safe_ln(log_func_v)
    #print step1
    step2 = (1-y) * safe_ln(1 - log_func_v)
    final = -step1 - step2
   
    return np.mean(final)
def grad_desc(theta_values, X, y, lr=.01, converge_change=.0001):"""The function calculates the theta_values for the given set of input vector
We converge when the change cost is the least"""
    #normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    s=(50,1)
    onesCol=np.ones(s)
    #X=np.array([1,X])
    #setup cost ite
    X=np.concatenate((onesCol,X), axis=1)
    cost_iter = []
    cost = cost_func(theta_values, X, y)

    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while(change_cost > converge_change):
        old_cost = cost
     
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))

        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1
     
    return theta_values,np.array(cost_iter)


a,b=grad_desc(theta_values,X,y)



print "this is final theta vector"
print a
print "this is cost at each iteration"
print b

def pred_values(theta, Xpred, hard=True):
    #normalize
    Xpred = (Xpred - np.mean(Xpred, axis=0)) / np.std(Xpred, axis=0)

    s=(50,1)
    onesCol=np.ones(s)

    Xpred=np.concatenate((onesCol,Xpred), axis=1)
    pred_prob =(logistic_func(theta, Xpred))
    #print pred_prob
    pred_value = np.where(pred_prob >= 0.5, 1, 0)
    if hard:
        return pred_value
    return pred_prob

s=pred_values(a,Xpred)
count=0
for i in range(0,50):
    if s[i]==int(y[i]):
        count+=1

print "Equation of hyperplane " + str(a[1])+"x + "+ str(a[2])+"y +"+str(a[3])+"z +"+str(a[0])+" = 0"

print str(count) + "  Points are classified"
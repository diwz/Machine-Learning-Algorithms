#FastMap Algorithm
import math
import random

import scipy

import matplotlib.pyplot as plt
from random import randint

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
K=2

res = np.zeros((len(dataPoints), K))

pivots = np.zeros((K, 2), "i")




def furthest(dist, o,col):
    for i in range(len(dist)):
       if dist[o][i]==max(dist[o]):
            break
    return i

def pickPivot(dist,col):#Find the two most distant points
    
    o1 = random.randint(0, len(dist) - 1)
    # print"hi"
    # print o1
    o2 = -1

    i=10

    while i > 0:
        o = furthest(dist,o1,col)
        if o == o2:
            break
        o2 = o
        o = furthest(dist,o2,col)
        if o == o1:
            break
        o1 = o
        i -= 1

    pivots[col] = (o1, o2)

    return (o1, o2)

def fastmap(dist,K,col):
    
    if K == 0:
        return 
    #else:
        
    px, py = pickPivot(dist,col)  
    #print "Picked %d, %d at K = %d" % (px, py, K)

    if distance(px, py, col,dist) == 0:
        for i in range(len(dist)):
            res[i][col] =0
        return

    for i in range(len(dist)):
        res[i][col] = cosine(i, px, py,col,dist)

    col+=1
    
    fastmap(dist,K - 1,col)

def cosine(i, x, y,col,dist):#Project the i'th point onto the line defined by x and y
    
    dix = distance(i, x,col,dist)
    diy = distance(i, y,col,dist)
    dxy = distance(x, y,col,dist)
    return (dix + dxy - diy) / 2 * math.sqrt(dxy)

def distance(x, y, k,dist):#Recursively compute the distance based on previous projections
    
    if k == 0:
        return abs(dist[x, y]) 

    rec = distance(x, y, k - 1,dist)
    resd = (res[x][k] - res[y][k]) 
    return abs(rec - resd)



def euclidean(x, y):#function to calculate euclidean distance
    return math.sqrt(sum((x - y) ** 2))

def distmatrix(dist,p):#function to calculate distance matrix
    
    for x in range(len(p)):
        for y in range(x, len(p)):
            if x == y:
                continue
            dist[x, y] = euclidean(p[x], p[y])
            dist[y, x] = dist[x, y]

    return dist


 

def final():
    points=a
    dist = np.zeros((len(a), len(a)))
    print "Distance matrix"
    dist = distmatrix(dist,points)
    print dist

    print "Mapping"
    if dist.max() > 1:
        dist /= dist.max()
    # print dist
    fastmap(dist, K,0)
    print "K = 2"
    
    print "Reduce Dimensions"
    print res*10




final()
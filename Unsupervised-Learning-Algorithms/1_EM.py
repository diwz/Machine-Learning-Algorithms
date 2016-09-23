#ExpectationMaximization
import matplotlib.pyplot as plt
from random import randint
import math
from numpy import matrix
import numpy as np
count=0

filename = 'clusters1.txt' #Read the data set training file
fileP = open(filename, 'rU')

temp = [r.split(',')  for r in fileP.read().split()]

dataPoints = [[]]
dataPoints =  [ [float(t[0]),float(t[1])] for t in temp]#type cast each point to float

X = np.array(dataPoints)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X=np.asmatrix(X)
print "Input Matrix:"
print X
centroidList = []
for i in range(0,3):
	centroidList.append((randint(0,len(dataPoints))))#pick 3 random indexes for initial centroids



newCentroidsList = []
for i in range(0,3):
	k=centroidList[i]
	newCentroidsList.append(dataPoints[k])
print "Intial means:"
print newCentroidsList


def newMatrix(newCentroidsList,k):#Function to initialize co variance matrices which calculates each element in the covariance matrix


	a=[]
	b=[]
	x11=sum([(dataPoints[i][0]-newCentroidsList[k][0])**2 for i in range(0,len(dataPoints))])
	x11= x11/(len(dataPoints))
	a.append(x11)
	
	x12=sum([(dataPoints[i][0]-newCentroidsList[k][0])*(dataPoints[i][1]-newCentroidsList[k][1]) for i in range(0,len(dataPoints)) ])
	x12= x12/(len(dataPoints))
	a.append(x12)

	x21=sum([(dataPoints[i][1]-newCentroidsList[k][1])*(dataPoints[i][0]-newCentroidsList[k][0]) for i in range(0,len(dataPoints)) ])
	x21= x21/(len(dataPoints))
	b.append(x21)

	x22=sum([(dataPoints[i][1]-newCentroidsList[k][1])**2 for i in range(0,len(dataPoints))])
	x22= x22/(len(dataPoints))
	b.append(x22)

	A = matrix([a,b])
	return A



def expFunc(mu,lisMat):#function for expectation step 
	
	finalE=[]
	for k in range(0,len(dataPoints)):
		a=[]
		e=[]
		for j in range(0,len(newCentroidsList)):
			l=np.asmatrix(lisMat[j])
			
			t=1.0/((2*np.pi*np.linalg.det(l))**0.5)
		
			y=(-0.5)*((X[k]- mu[j])*l.I*(X[k]- mu[j]).T)
			a.append(y)
			e.append(t*math.exp(y))
	
		finalE.append(e)
	
	fE= np.array(finalE)
	
	P_Cx=fE/fE.sum(axis=0)
	#print P_Cx
	return P_Cx


def maximization(P_Cx):#function for maximization step
	temp=[]
	for i in range(0,3):
		temp.append((P_Cx[0:len(dataPoints),i] *X)/(sum(P_Cx[0:len(dataPoints),i])*150 ))


	temp=np.array(temp)
	temp1=[]
	for i in range(0,3):
		temp1.append(temp[i][0])
	temp1=np.array(temp1)
	#print temp1
	
	lisMat=[]
	
	for k in range(0,3):
		a=[]
 		b=[]		
		x11=sum([P_Cx[0:len(dataPoints),k][i]*(dataPoints[i][0]-mu[k][0])**2 for i in range(0,len(dataPoints))])
		x11= x11/(len(dataPoints))
		#print x11
		a.append(x11)
		x12=sum([P_Cx[0:len(dataPoints),k][i]*(dataPoints[i][0]-mu[k][0])*(dataPoints[i][1]-mu[k][1]) for i in range(0,len(dataPoints)) ])
		x12= x12/(len(dataPoints))
		#print x12

		a.append(x12)

		x21=sum([P_Cx[0:len(dataPoints),k][i]*(dataPoints[i][1]-mu[k][1])*(dataPoints[i][0]-mu[k][0]) for i in range(0,len(dataPoints)) ])
		x21= x21/(len(dataPoints))
		
		
		b.append(x21)
		x22=sum([P_Cx[0:len(dataPoints),k][i]*(dataPoints[i][1]-mu[k][1])**2 for i in range(0,len(dataPoints))])
		x22= x22/(len(dataPoints))
		b.append(x22)
		
		s = np.array([a,b])
		#print s
		#A=np.asmatrix(s)
		lisMat.append(np.array([a,b]))
	return temp1,lisMat


def em(X,mu,coMatrix):#em algorithm

	global count
	count+=1
	
	mutemp=mu
	# print "run"+str(count)
	# print coMatrix
	ResExp=expFunc(mutemp,coMatrix)
	muCentroids,coMatrix=maximization(ResExp)
	#print coMatrix
	if np.array_equal(muCentroids, mutemp) or count==10:
		# print "hete:"
		print "Final Means with Covariance Matrices:"
		for i in range(3):
 			print "Mean "+str(i+1)
 			print muCentroids[i]
 			print "Covariance Matrix"
 			print coMatrix[i]
		
		return muCentroids,coMatrix
	
	em(X,muCentroids,coMatrix)

mu=np.array(newCentroidsList)


lisMat1=[]
for i in range(0,3):
	A=newMatrix(newCentroidsList,i)
	lisMat1.append(A)

em(X,mu,lisMat1)


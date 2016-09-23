#Perceptron Learning ALgorithm
import matplotlib.pyplot as plt
from random import randint
import math as m
import numpy as np
unit_step = lambda x: 1 if x >= 0 else -1
rate=0.001
theta=0
filename = 'linear.txt' #Read the data set training file
fileP = open(filename, 'rU')

temp = [r.split(',')  for r in fileP.read().split()]
dataPoints = [[]]
dataPoints =  [ [float(t[0]),float(t[1]),float(t[2])] for t in temp]#type cast each point to float

labels =[ float(t[3]) for t in temp]

# print dataPoints
# print labels

weights = []

for k  in range(0,4):
  	weights.append(0)

iteration=0
def calOutput(theta,weights,x,y,z):
	sum = x*weights[0]+y*weights[1]+z*weights[2]+weights[3]
	return unit_step(sum)


def perceptron():
	iteration=0
	
	while True:
		
		globalError=0
		iteration +=1
		error_count=0
		#print "hi"
		for i in range(0,len(dataPoints)):
			#print "iteration" + str(i)
			output= calOutput(theta,weights,dataPoints[i][0],dataPoints[i][1],dataPoints[i][2])
			
			localError = labels[i] -output
			if localError != 0:
				error_count += 1
								
				for k in range(3):
					weights[k] += rate*localError*dataPoints[i][k]

				weights[3]+=rate*localError
			

		if (error_count==0 ):
			print "Number of Iterations " + str(iteration)
			break



perceptron()
print "Final weight vector " +str(weights)	
print "Equation of hyperplane " + str(weights[0])+"x + "+ str(weights[1])+"y +"+str(weights[2])+"z +"+str(weights[3])+" = 0"


count=0
for i in range(0,50):
	x=dataPoints[i][0]*weights[0]+dataPoints[i][1]*weights[1]+dataPoints[i][2]*weights[2]+weights[3]
	if unit_step(x) == labels[i]:
		
		print "match"
		count+=1
	else:

		print "not "
print count



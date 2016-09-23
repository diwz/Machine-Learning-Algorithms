import matplotlib.pyplot as plt
from random import randint
import math

filename = 'km-data.txt' #Read the data set training file
fileP = open(filename, 'rU')

temp = [r.split(',')  for r in fileP.read().split()]
dataPoints = [[]]
dataPoints =  [ [float(t[0]),float(t[1])] for t in temp]#type cast each point to float

centroidList = []
for i in range(0,3):
	centroidList.append((randint(0,len(dataPoints))))#pick 3 random indexes for initial centroids

# print centroidList
# print dataPoints

def plotter(dataPoints,cList,flag):
	x_list = []
	y_list = []
	for dp in dataPoints:
		x_list.append(dp[0])
		y_list.append(dp[1])

	plt.plot(x_list,y_list,'ro')
	for i in range(0,3):
		if flag ==  0:
			plt.plot(dataPoints[centroidList[i]][0],dataPoints[centroidList[i]][1],'g^',label="Intial",markersize=10)
		else:
		    plt.plot(cList[i][0],cList[i][1],'g^',markersize=10)


	plt.axis([min(x_list), max(x_list), min(y_list), max(y_list)])
	plt.xlabel("X axis")
	plt.ylabel("Y axis")
	if flag ==  0:
		plt.suptitle("Initial Assumed Centroids with Data Points")
	else:
		plt.suptitle("Centroids after K-means ")
	plt.show()

plotter(dataPoints,centroidList,0)
flag = raw_input("Which distance to use for calculation 1.)L1/Manhattan 2.)L2/Eucidean ")

def distL2(x1,x2,y1,y2,flag):#calculate the euclidean distance
	if(flag == 1):
		return (math.fabs(x2-x1)+math.fabs(y2-y1))	
	else:
		return math.sqrt((x2-x1)**2.0+(y2-y1)**2.0)


def initializeClusters(dataPoints,centroidList):
	clusters = {centroidList[0]:[],centroidList[1]:[],centroidList[2]:[]}
	k=0
	
	
	for point in dataPoints:
		temp1 = []
		distList = []
		for i in range(0,3):
			distList.append(distL2(point[0],dataPoints[centroidList[i]][0],point[1],dataPoints[centroidList[i]][1],flag))
		k=distList.index(min(distList))		     
		temp1.append(point[0])
		temp1.append(point[1])
		clusters[centroidList[k]].append(temp1)

	return clusters

                      
	
cluster = initializeClusters(dataPoints,centroidList)


def initialCentroids(clusters):
	newCentroidsList=[]
	
	for k in range(0,3):
		x_new = sum([clusters[centroidList[k]][i][0] for i in range(0,len(clusters[centroidList[k]]))])/len(clusters[centroidList[k]])
		y_new = sum([clusters[centroidList[k]][i][1] for i in range(0,len(clusters[centroidList[k]]))])/len(clusters[centroidList[k]])
		newCentroidsList.append([x_new,y_new])
	return newCentroidsList

newCentroidsList= initialCentroids(cluster)
print newCentroidsList
def newCentroidsKmeans(clusters):
	newCentroidsList=[]
	for k in range(0,3):
		x_new = sum([clusters[k][i][0] for i in range(0,len(clusters[k]))])/len(clusters[k])
		y_new = sum([clusters[k][i][1] for i in range(0,len(clusters[k]))])/len(clusters[k])
		newCentroidsList.append([x_new,y_new])
	return newCentroidsList
		

def kMeans(dataPoints,newCentroidsList):
	clusters = {0:[],1:[],2:[]}
	k=0
	
	tempList = newCentroidsList
	
	for point in dataPoints:
		temp1 = []
		distList = []
		for i in range(0,3):
			distList.append(distL2(point[0],newCentroidsList[i][0],point[1],newCentroidsList[i][1],flag))
		k=distList.index(min(distList))  
		temp1.append(point[0])
		temp1.append(point[1])
		clusters[k].append(temp1)

	cList=newCentroidsKmeans(clusters)
	
	if cmp(tempList,cList) == 0:
		
		for i in range(0,3):
			print "Centroid " + str(i+1) + str(cList[i])
			print clusters[i]
			print "\n"
		plotter(dataPoints,cList,1)
		return

	kMeans(dataPoints,cList)

kMeans(dataPoints,newCentroidsList)






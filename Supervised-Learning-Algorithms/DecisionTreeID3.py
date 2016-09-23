
filename = 'dt-data.txt'#Read the data set training file
fileP = open(filename, 'rU')# Open in file handler

data = [(line.strip()).split(',') for line in fileP]
fileP.close()

class Tree: #class defination of tree node object
	 def __init__(root, parent=None):
	  root.parent = parent
	  root.children = []
	  root.label = None
	  root.splitFeatureValue = None
	  root.splitFeature = None

def cleandata(data): # function for formating the data set and removing the noisy data parts from the text file
	for i in range(2,len(data)):
		temp=data[i][0].split()#Stripping the row number in each record ex:'01 Large' , == > 'Large',
		data[i][0]=temp[1]
		data[i][7]=data[i][7].strip(';')
	data[0][0]= data[0][0].strip('(')
	data[0][7]= data[0][7].strip(')')
	return data[2:len(data)],data[0]


d1,attrList=cleandata(data)# d1 is the main data set, attrList is the list of features


def uniquecounts(dataSubset):
	"""Function to count the class-label values,it takes the 
	data set as input and returns a dictionary of counts of the class label"""
	results={}
	for row in dataSubset:
	        # The result is the last column ie Yes/No from the class-label 'Enjoy'
	        r=row[len(row)-1]
	        if r not in results: results[r]=0
	        results[r]+=1
	return results



def entropy(dataSubset):
	"""Function calculates the entropy for the data subset"""
	from math import log

	log2=lambda x:log(x)/log(2)  
	# Now calculate the entropy
	entropy=0.0
	results = uniquecounts(dataSubset)
	for r in results.keys():
	# current probability of class
		p=float(results[r])/len(dataSubset) 
		entropy=entropy-p*log2(p)
	return entropy
	




def splitData(data, featureIndex):
   '''Function to iterate over the subsets of data corresponding to each value
       of the feature at the index featureIndex. '''
 
   # get possible values of the given feature
   attrValues = [row[featureIndex] for row in data]
   attr = set(attrValues)#use set object to get only unique values of that feature
   dataSubset = [] #A list containing data sets as per the split on each feature value
   for aValue in attr:#for each unique value
   		
		dataSubset.append([row for row in data if row[featureIndex] == aValue])
  
   return dataSubset



def gain(dataSubset, featureIndex):
   ''' Function to compute the expected gain from splitting the data along all possible
       values of feature. '''
   entropyGain = entropy(dataSubset)#
   
   for dataSubset_Sub in splitData(dataSubset, featureIndex):
      entropyGain -= entropy(dataSubset_Sub)
      
   return entropyGain



def baseCriteria(dataSubset, node):
   ''' Function to get label node with the majority of the class labels values in the given data set  '''
   labels = [d[7] for d in dataSubset]
   classLabelValue = max(set(labels), key=labels.count)
   node.label = classLabelValue 

   return node


def buildDecisionTree(data, root, remainingFeatures):
   ''' Build a decision tree from the given data, appending the children
       to the given root node (which may be the root of a subtree). '''

   if len(remainingFeatures) == 0:#Base criteria to return from loop when there are no remaining features to split on
      return baseCriteria(data, root)

   # find the index of the best feature to split on
   bestFeature = max(remainingFeatures, key=lambda index: gain(data, index))
 
   if gain(data, bestFeature) == 0:
      return baseCriteria(data, root)

   root.splitFeature = bestFeature
   # add child nodes and process recursively
   for dataSubset in splitData(data, bestFeature):
      aChild = Tree(parent=root)
      aChild.splitFeatureValue = dataSubset[0][bestFeature]
      root.children.append(aChild)

      buildDecisionTree(dataSubset, aChild, remainingFeatures - set([bestFeature]))

   return root





def printDecisionTree(root, indent=""):
 

   if root.children == [] :#print the class label
    print "%s%s, %s " % (indent, root.splitFeatureValue, root.label)
   else:
      printDecisionTree(root.children[0], indent + "\t ")

      if indent == "": 
         print "%s%s" % (indent,attrList[root.splitFeature])
      else:
         print "%s%s, %s" % (indent, root.splitFeatureValue, attrList[root.splitFeature])

   	
      for i in range(1,len(root.children)):

      	printDecisionTree(root.children[i], indent + "\t ") 





tree = buildDecisionTree(d1, Tree(), set(range(len(d1[0])-1)))
Query = ['Large', ' Moderate', ' Cheap', ' Loud', ' City-Center', ' No', ' No']

printDecisionTree(tree)

def predictor(root,q):

   ''' Predicting the class label of a Query by traversing the given decision tree. '''
   if root.children == []:
      return root.label
   else:
      subChildNode = [child for child in root.children
         if child.splitFeatureValue == q[root.splitFeature]]

      
      return predictor(subChildNode[0], q)

print "\n The Query posed:"
print attrList[0:7]
print Query

if predictor(tree,Query) == ' Yes':
   print "Yes you will enjoy the night-out!"
else:
   print "Sorry you wont enjoy the night out!"



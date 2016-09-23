import numpy as np
unit_step=lambda x: 1 if x >=0 else -1

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2
layers=[2,2,1]  

        
weights = []# Set weights
     
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
for i in range(1, len(layers) - 1):
    r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
    weights.append(r)
# output layer - random((2+1, 1)) : 3 x 1
r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
weights.append(r)

def feedForward( X, y, learning_rate=0.0001, epochs=10000):
    # Add column of ones to X
    # This is to add the bias unit to the input layer
    ones = np.atleast_2d(np.ones(X.shape[0]))
    X = np.concatenate((ones.T, X), axis=1)

    for k in range(epochs):
        i = np.random.randint(X.shape[0])
        a = [X[i]]
        for l in range(len(weights)):
                dot_value = np.dot(a[l], weights[l])
                tan = tanh(dot_value)
                a.append(tan)
        # output layer
        error = y[i] - a[-1]
        deltas = [error * tanh_prime(a[-1])]

        # we need to begin at the second to last layer 
        # (a layer before the output layer)
        for l in range(len(a) - 2, 0, -1): 
            deltas.append(deltas[-1].dot(weights[l].T)*tanh_prime(a[l]))

        # reverse
        # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
        deltas.reverse()

        # backpropagation
        # 1. Multiply its output delta and input tanh 
        #    to get the gradient of the weight.
        # 2. Subtract a ratio (percentage) of the gradient from the weight.
        for i in range(len(weights)):
            layer = np.atleast_2d(a[i])
            delta = np.atleast_2d(deltas[i])
            weights[i] += learning_rate * layer.T.dot(delta)

    

def predict( x): 
    a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
    for l in range(0, len(weights)):
        a = tanh(np.dot(a, weights[l]))
    return a

if __name__ == '__main__':


    fileP = open("nnsvm-data.txt","rU")
    temp = [r.split(' ')  for r in fileP.read().split('\n')]
    dataPoints = [[]]
    del(temp[100])
    dataPoints =  [ [float(t[0]),float(t[1])] for t in temp]
    label=[float(t[2]) for t in temp]
    X=np.array(dataPoints)
    #print X
    y=np.array(label)
    count=0
    feedForward(X[46:71],y[46:71])
    print "Weights in the Layers"
    print (weights[0]) 
    print "Final Weights"  
    print weights[1] 
    for i in range(0,46):
       #print(X[i],y[i], unit_step(predict(X[i])) )
       if(y[i] == float(unit_step(predict(X[i]))) ):
    	count+=1

    for i in range(71,100):
       #print(X[i],y[i],unit_step(predict(X[i])))
       if(y[i] == float(unit_step(predict(X[i]))) ):
    	count+=1

    print "Accuracy:"
    print (count/75.0)*100

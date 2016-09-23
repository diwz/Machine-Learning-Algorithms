import math
import numpy as np
import matplotlib.pyplot as plt
filename = 'hmm-data.txt' #Read the data set training file
fileP = open(filename, 'rU')
temp = [r.split(',')  for r in fileP.read().split('\n')]
dataPoints =  [ t for t in temp[2:13]]
dataPoints1 =  [ t for t in temp[24:36]]
dataP=[]
for k in range(0,10 ):
	p=dataPoints[k][0].split()
	dataP.append(map(int,p))
noisyTimeStamp=[]

for k in range(0,11):
	pfloat=dataPoints1[k][0].split()
	noisyTimeStamp.append(map(float,pfloat))

def get_directions( dataP , i, j):#we find the available positions where the robot can move
	directions = []
	
	if  dataP[i][j] == 0:
		return directions	
	if 0 < j <=9: 		
		if  dataP[i][j-1] == 1:
			directions.append([i,j-1])	
	if 0 <=j < 9:	
		if  dataP[i][j+1] == 1:
			directions.append([i,j+1])	 
	if 0 < i <=9: 		
		if  dataP[i-1][j] == 1:
			directions.append([i-1,j])
	if 0 <= i < 9: 		
		if  dataP[i+1][j] == 1:
			directions.append([i+1,j])
	return directions	
	
def compute_cpt(tower,n):# compute the cpt for each tower state
	t1=[]
	i=1
	while i <= n:
		cp_state =[0 for k in range(0,int(max_dist*10))]
	
		state = fs_dict[i]
		distance = math.sqrt(((state[0] - tower[0])* (state[0] - tower[0])) +((state[1] - tower[1])* (state[1] - tower[1])))
	
		range1 = round(0.7*distance , 1)
		range2 = round(1.3*distance , 1)
		length_of_range = ((range2-range1) + 0.1)*10

		probability = float(1.0/length_of_range)

		k = range1
		while k <= range2 and k<=max_dist :
			cp_state[noisyTimeStamp_mapping[k]-1] = probability
			k = k+0.1
			k = round(k,1)
		
		t1.append(cp_state)
		i = i + 1
	return t1

count_freeStates = 0
fs = []
fs_label = {}
fs_dict = {}
for i in range(0,len(dataP)):
	for j in range(0,len(dataP)):
		state = []
		if( dataP[i][j] == 1):
			count_freeStates = count_freeStates + 1
			state.append(i)
			state.append(j)
			fs.append(state)	
				
i = 1
for free_state in fs:
	fs_label[tuple(free_state)] = i
	fs_dict[i] = free_state
	i +=1


transition_matrix = []	
each_state_transition = []

for i in range(0,len(dataP)):
	for j in range(0,len(dataP)):
		if  dataP[i][j] == 1:
			directions = get_directions( dataP , i, j )
			num_of_directions = float(len(directions))
			probability = float(1.0/num_of_directions)
			each_state_transition = [0 for k in range(0,count_freeStates)]
			
			for direct in directions:
				x = direct[0]
				y = direct[1]
				state_number = fs_label[(x,y)]
				each_state_transition[state_number - 1] = probability
			transition_matrix.append(each_state_transition)
	
precision = 0.1
max_dist = 14.1
noisyTimeStamp_mapping = {}
i = 0
count = 0
while i <= max_dist:
	noisyTimeStamp_mapping[i] = count
	count = count +1
	i = i + 0.1
	i = round(i,1)

t1=compute_cpt([0,0],count_freeStates)
t2=compute_cpt([0,9],count_freeStates)
t3=compute_cpt([9,0],count_freeStates)
t4=compute_cpt([9,9],count_freeStates)

time_total = len(noisyTimeStamp)
states_total = len(fs)
prob = [[0 for k in range(0,time_total)] for j in range(0,len(fs))]
back = [[0 for k in range(0,time_total)] for j in range(0,len(fs))]
time = 0 
state = 0
fs_dict[-1] = [10,10]
fs_dict[0] = [10,10]
while state < 87:
	n1 = int(round(noisyTimeStamp[0][0],1)*10)
	n2 = int(round(noisyTimeStamp[0][1],1)*10)
	n3 = int(round(noisyTimeStamp[0][2],1)*10)
	n4 = int(round(noisyTimeStamp[0][3],1)*10)
	prob[state][0] = float(1.0/87.0) * t1[state][n1] * t2[state][n2] * t3[state][n3] * t4[state][n4]  
	coordinates = fs_dict[state+1]
	directions = get_directions( dataP , coordinates[0], coordinates[1])
	state_0 = fs_label[(directions[0][0],directions[0][1])]
	back[state][0] = state_0
	state = state + 1
time = 1 
while time < 11:
	state  = 0
	while state < states_total:
		n1 = int(round(noisyTimeStamp[time][0],1)*10)
		n2 = int(round(noisyTimeStamp[time][1],1)*10)
		n3 = int(round(noisyTimeStamp[time][2],1)*10)
		n4 = int(round(noisyTimeStamp[time][3],1)*10)
		previous_time = time - 1
		x = 0
		maximum = 0.0
		back_value = -1
		while x < states_total:
			value = prob[x][previous_time]*transition_matrix[x][state]*t1[state][n1] * t2[state][n2] * t3[state][n3] * t4[state][n4]
			if value > maximum:
				maximum = max(maximum,value)
			x = x + 1
		prob[state][time] = maximum
		maximum = 0.0
		back_value = -1
		x = 0
		while x < states_total:
			value = prob[x][previous_time]*transition_matrix[x][state]
			if value > maximum:
				maximum = max(maximum,value)
				back_value = x
			x = x + 1
		back[state][time] = back_value
		state = state +1
	time = time+1

path = {}
final_states = []
maximum = 0.0
state = 0 
final_state = state
while state < 87:
	if prob[state][10] > maximum:
		maximum = prob[state][10]
		final_state = state
	state = state + 1
final_states.append(fs_dict[final_state+1])
path[11] = fs_dict[final_state+1]
i = 10
while i > 0:		
	back_state = back[final_state][i]
	path[i] = fs_dict[back_state+1]
	final_state = back_state
	i = i - 1
print path
coor=path.values()
x=[]
y=[]
for c in coor:
	x.append(c[0])
	y.append(c[1])
fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(1,10,1))
ax.set_yticks(np.arange(1,10,1))
plt.grid()
plt.axis([0, 10, 0, 10])
plt.plot(x, y, marker='o', linestyle='--', color='r', label='Path')
plt.plot(x[0],y[0],marker='o',color='g',label="Start",markersize=10)
plt.plot(x[10],y[10],marker='o',color='b',label="Finish",markersize=10)
plt.title('Robot Grid')
plt.legend()
plt.show()
	

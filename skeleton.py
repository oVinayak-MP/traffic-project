import random
import numpy as np
from collections import deque
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.layers import Dense,Flatten,Concatenate,Input,concatenate
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras import backend as K
from random import *

import random
import time
import os, sys
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:   
     sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
from tqdm import tqdm

EPISODES = 5000
newTimeStamp=0


cmd = ['sumo-gui', '-c', 'newjunction.sumocfg','--waiting-time-memory','10','-e','500']


class DQNAgent:
  def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95	#discount rate
        self.epsilon = 1.0	#exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()        
  def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

	

		
	'''
	posnet = Sequential()
	posnet.add(Conv2D(16, kernel_size=(4, 4), strides=(2, 2), #First layer
                 activation='relu',	#Change activation function to nonlinear
                 input_shape=(1,lane_length,lane_width)))
	posnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))	#Not sure
	posnet.add(Conv2D(32, (2, 2), activation='relu'))
	posnet.add(MaxPooling2D(pool_size=(2, 2)))
	posnet.add(Flatten())
	
	boolnet = Sequential()
	boolnet.add(Conv2D(16, kernel_size=(4, 4), strides=(2, 2), #First layer
                 activation='relu',	#Change activation function to nonlinear
                 input_shape=(1,lane_length,lane_width)))
	boolnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))	#Not sure
	boolnet.add(Conv2D(32, (2, 2), activation='relu'))
	boolnet.add(MaxPooling2D(pool_size=(2, 2)))
	boolnet.add(Flatten())
	
	combine=Concatenate([posnet,boolnet])
	
	combine.add(Dense(1000, activation='relu'))
	combine.add(Dense(num_classes, activation='softmax'))

	combine.add() #fully connected
	combine.add() #fully connected
	#maybe use optimizer rmsprop
	combine.compile(loss=self._huber_loss,optimizer=Adam(lr=self.learning_rate))'''
	
 
  def _build_model(self):
        first_con_input = Input(shape=(16,20,1))    #size of matrix
        second_con_input = Input(shape=(16,20,1))   #size of matrix
        third_et_input = Input(shape=(20,))
        model1_1=Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid',data_format='channels_last',padding='same')(first_con_input)
        model2_1=Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid',data_format='channels_last',padding='same')(second_con_input)
        model1_2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model1_1)	#
        model2_2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model2_1)
        model1_3=Conv2D(32, (2, 2), activation='relu')(model1_2)
        model2_3=Conv2D(32, (2, 2), activation='relu')(model2_2)
        model1_4=MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(model1_3)	#
        model2_4=MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(model2_3)	#
        model1_5=Flatten()(model1_4)
        model2_5=Flatten()(model2_4)
        model1_6=Dense(256, activation='relu')(model1_5)
        model2_6=Dense(256, activation='relu')(model2_5)
        model3_1=Dense(20, activation='relu')(third_et_input) #size of state
        #model3_2=Flatten()(model3_1)  #not used
        tempmodel=concatenate([model1_6,model2_6])
        finalemodel_1=concatenate([tempmodel,model3_1],axis=1)
        finalemodel_2=Dense(532, activation='relu')(finalemodel_1)
        finalmodel=Dense(20, activation='softmax')(finalemodel_2) #output row and column
        final=Model(inputs=[first_con_input,second_con_input,third_et_input],outputs=[finalmodel])
        final.compile(loss='categorical_crossentropy',optimizer=Adam(lr=self.learning_rate),metrics=['accuracy'])
        '''if self.debug == True:
            print ("The network model")
            final.summary()'''
        return final



  def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

  def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
  	if np.random.rand() <= self.epsilon:
  		return random.randrange(self.action_size)
  	act_values = self.model.predict(state)
  	return np.argmax(act_values[0])  # returns action, check output form of cnn

  def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
        
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
        self.model.save_weights(name)




#######calculate immediate reward from cumulative delay####

oldTimeStamp=0
elapsedTime=0 #initial arbitrary
def Reward():
	W=0
	global elapsedTime
	for lane in laneid:
		for id in traci.lane.getLastStepVehicleIDs(lane):
			waitingTime=traci.vehicle.getWaitingTime(id)		
			if(waitingTime<elapsedTime):
				W+=waitingTime
			else:
				W+=elapsedTime
	return W
#############end of reward calculation function#########################

def performAction(actionIndex):
	global newTimeStamp
	global oldTimeStamp
	global elapsedTime
	
	newTimeStamp=time.time()
	traci.trafficlights.setRedYellowGreenState('5', actions[actionIndex])
	elapsedTime=newTimeStamp-oldTimeStamp
	oldTimeStamp=newTimeStamp
	print "Action Taken: ",actions[actionIndex]
	traci.simulationStep()
	currentStateToMatrix() #new matrices
	return 
#####action#########

def actionToNumber(action):
	num=[0 for x in range(20)]
	for i in range(action.length-1):
		if(action[i]=='g'):
			num[i]=1
		elif(actio[i]=='G'):
			num[i]=2
		elif (num[i]=='Y'):
			num[i]=3
			
	return num
	
###join pos and bool matrices to a type state and return
def matricesToState():
	currentStateToMatrix()
	return P,V

##########end of matricestostate##############


speedlimit=50 													##############whatever is the speed limit in correct units
avgvl=5

laneid=['w2e_s_3','w2e_s_2','w2e_s_1','w2e_s_0',
	's2n_s_3','s2n_s_2','s2n_s_1','s2n_s_0',
	'e2w_s_3','e2w_s_2','e2w_s_1','e2w_s_0',
	'n2s_s_0','n2s_s_1','n2s_s_2','n2s_s_3']
	
row=	[3,2,1,0,
	 15,14,13,12,
	 7,6,5,4,
	 11,10,9,8]	
	 
P = [[0 for x in range(21)] for y in range(17)]
V = [[0 for x in range(21)] for y in range(17)]


def currentStateToMatrix():
	avgvl=5
	for lane in laneid:
		for id in traci.lane.getLastStepVehicleIDs(lane):
			pos=traci.vehicle.getLanePosition(id)
			colIndex=int(pos/avgvl)			
			index=laneid.index(lane)
			rowIndex=row[index]
			
			if(colIndex<20):			
				P[rowIndex][colIndex]=1
				V[rowIndex][colIndex]=speed=traci.vehicle.getSpeed(id)				
						
			
	return


actions = [
"GGGGgrrrrrGGGGgrrrrr",
"yyyygrrrrryyyygrrrrr",
"rrrrGrrrrrrrrrGrrrrr",
"rrrryrrrrrrrrryrrrrr",
"rrrrrGGGGgrrrrrGGGGg",
"rrrrryyyygrrrrryyyyg",
"rrrrrrrrrGrrrrrrrrrG",
"rrrrrrrrryrrrrrrrrry",
]

def printV():
	for i in range(16):
    			print ''
    			for j in range(20):
    				print int(V[i][j]),
    	return
    	
def printP():
	for i in range(16):
    			print ''
    			for j in range(20):
    				print int(P[i][j]),    	
    				
	return
	
	    				
if __name__ == "__main__":
	
		newTimeStamp=0
		oldTimeStamp=0
		elapsedTime=0 #initial arbitrary
		state_size = 16# state size understandable by cnn 2-d matrix?
		action_size = 8 # check sumo traffic light phases
		agent = DQNAgent(state_size, action_size)
		done = False
		batch_size = 32 #use size in paper
		traci.start(cmd)	#start simulation
		oldTimeStamp=time.time()
		'''for i in tqdm(range(500)):		
				performAction(randint(0,7))
				print "reward=",reward()
				#matricesToState()
				currentStateToMatrix()	
				printV()    		
		traci.close()
		time_end = time()'''
		for e in range(EPISODES):
				firstaction=randint(0,7)
				performAction(firstaction) #Initial Random Action 
				matricesToState()     
				state = {"input_1": np.array(P),"input_2":np.array(V),"input_3":actions[firstaction]}
				for Time in range(500):   #time range?
					action = agent.act(state)
					performAction(action)	
					matricesToState()	  
					actionState=[ 0 for x in range(8)]
					actionState[action]=1     
					next_state = {"input_1": np.array(P),"input_2":np.array(V),"input_3":np.array(actionState)}
					reward=Reward()
          #reward = reward if not done else -10
					done=0
					agent.remember(state, action, reward, next_state, done) #make sure state,action,next_state parameters cnn can understand
					state = next_state
					'''if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break'''
					if len(agent.memory) > batch_size:
						agent.replay(batch_size)
					# if e % 10 == 0:
					#     agent.save("./save/cartpole-ddqn.h5"
		traci.close()
		time_end = time()
        

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 5000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
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
	combine.compile(loss=self._huber_loss,optimizer=Adam(lr=self.learning_rate))
	
 
  def _build_model(self):
        # Neural Net for Deep-Q learning Model
        posnet = Sequential()
				posnet.add(Convolution2D(16, 4, 4, activation='relu', input_shape=(1,lane_length,lane_width)))

       # posnet.add(Dense(24, input_dim=self.state_size, activation='relu'))
        posnet.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss=self._huber_loss,optimizer=Adam(lr=self.learning_rate))
        return model
'''


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
def reward():

#############end of reward calculation function#########################

########action=NSG,EWG,NSLG,EWLG######
def performaction(action):
	actionstring=#corresponding light signals for NSG,EWG,NSLG,EWLG
	setRedYellowGreenState(tlsID, actionstring[action])
	traci.simulationStep()
	currentstatetomatrix() #new matrices
	return matricestostate()
#####action#########


###join pos and bool matrices to a type state and return
def matricestostate():


##########end of matricestostate##############


speedlimit=50 																		##############whatever is the speed limit in correct units
def currentstatetomatrix():

	for id in traci.vehicle.getIDList():
			
			    pos =  traci.vehicle.getPosition(id)[1]-251
			    #n1 = int((400-250)/traci.vehicle.getLength(id))
			    indexpos = int(pos/traci.vehicle.getLength(id))
			    lane=traci.vehicle.getLaneID(id)
			    if indexpos<12:
				    if 		(lane =="n_0_0"):
				    	north[indexpos][0]=1
					    northSpeed[n2][0]= traci.vehicle.getSpeed(id)/speedlimit
						else if (lane =="xxxxx"):	###############################lane id	for next lane
							north[indexpos][1]=1
							northSpeed[n2][0]= traci.vehicle.getSpeed(id)/speedlimit
						else if (lane =="xxxxx"):	###############################lane id	for next lane
							north[indexpos][2]=1
							northSpeed[indexpos][0]= traci.vehicle.getSpeed(id)/speedlimit
						else if (lane =="xxxxx"):	###############################lane id	for next lane
							north[indexpos][3]=1
							northSpeed[indexpos][0]= traci.vehicle.getSpeed(id)/speedlimit


if __name__ == "__main__":
    state_size = # state size understandable by cnn 2-d matrix?
    action_size = 4 # NSG,EWG,NSLG,EWLG
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32 #use size in paper

		traci.start(cmd)	#start simulation
		
    for e in range(EPISODES):
        traci.trafficlights.setRedYellowGreenState("0", random_action())
        currentstatetomatrix()
        state = matricestostate()
        for time in range(500):   #time range?
            action = agent.act(state)
            
            next_state=performaction(action)
            reward=reward()
            # done, _ = env.step(action)
            reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done) #make sure state,action,next_state parameters cnn can understand
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")




import random
import numpy as np
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Concatenate,Input,concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.convolutional import Conv2D,MaxPooling2D


cmd = ['sumo-gui', '-c', '../sumo-map/sumo-map.sumocfg','--waiting-time-memory','10','-e','500']

import os, sys
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
import traci
from tqdm import tqdm


class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size=state_size
        self.action_size=action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.00
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.debug = True
        #self.traci=traci
        self.avgvl=5
        self.lanearrylength=50
        self.maxspeed=15

        self.lanelist=['1to5_0','1to5_1','1to5_2','1to5_3','2to5_0','2to5_1','2to5_2','2to5_3','3to5_0','3to5_1','3to5_2','3to5_3','4to5_0','4to5_1','4to5_2','4to5_3']#create a general function for it


    def printd(self,str):
        if self.debug == True:
            print str


    def getSpeedandPosition(self,traci,laneid,debug):
        #This function returns speed and position for the give lane it also should contains average vehicle length

        lanepos=[0]*self.lanearrylength
        lanespeed=[0]*self.lanearrylength
        cnt=0
        if (debug>=1):
             print "lane id:" + str(laneid)
        for id in traci.lane.getLastStepVehicleIDs(laneid):
            speed=traci.vehicle.getSpeed(id)
            pos=traci.vehicle.getLanePosition(id)
            pos=int(pos/self.avgvl)
            cnt=cnt+1
            lanepos[pos]=1
            lanespeed[pos]=traci.vehicle.getSpeed(id)

            if (debug >= 1):
                    print "index:" + str(pos)
                    print "speed:" + str(lanespeed[pos])

        if(debug>=1):
             print "number of vehicles:" + str(cnt) +"\n\n"
        lanespeed[:]=[round(x/self.maxspeed,2) for x in lanespeed]
        return lanespeed,lanepos
    def starttraci(self):
        traci.start(cmd)
        return

    def getStateMat(self,traci,lanelist,debug):
        speedl=[]
        posl=[]
        for x in lanelist:
             speedi,posi=self.getSpeedandPosition(traci,x,debug-1)
             speedl.append(speedi)
             posl.append(posi)
        print "lane l" + str(len(speedl)) + " " + str(len(lanelist))
        return speedl,posl


    def create_model(self):                          #creates the model using fumctional API
        first_con_input = Input(shape=(12,20,1))    #size of matrix
        second_con_input = Input(shape=(12,20,1))   #size of matrix
        third_et_input = Input(shape=(8,))
        model1_1=Conv2D(16, kernel_size=(4, 4), strides=(2, 2), activation='sigmoid',data_format='channels_last',padding='same')(first_con_input)
        model2_1=Conv2D(16, kernel_size=(4, 4), strides=(2, 2), activation='sigmoid',data_format='channels_last',padding='same')(second_con_input)
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
        model3_1=Dense(18, activation='relu')(third_et_input) #size of state
        #model3_2=Flatten()(model3_1)  #not used
        tempmodel=concatenate([model1_6,model2_6])
        finalemodel_1=concatenate([tempmodel,model3_1],axis=1)
        finalemodel_2=Dense(530, activation='relu')(finalemodel_1)
        finalmodel=Dense(8, activation='softmax')(finalemodel_2) #output row and column
        final=Model(inputs=[first_con_input,second_con_input,third_et_input],outputs=[finalmodel])
        final.compile(loss='categorical_crossentropy',optimizer=Adam(lr=self.learning_rate),metrics=['accuracy'])
        if self.debug == True:
            print ("The network model")
            final.summary()
        return final



a = DQNAgent(100,200)
a.starttraci()
#a.getSpeedandPosition(traci,)
while True:
    traci.simulationStep()
    a.getStateMat(traci,a.lanelist,1)

'''
mod=a.create_model()
pos= np.random.rand(10000,12,20,1)
#pos= np.random.rand(12,20)
#pos=np.reshape(pos,(1,12,20,1))
#speed= np.random.rand(12,20)
#speed= np.reshape(speed,(1,12,20,1))
speed= np.random.rand(10000,12,20,1)
state = np.random.rand(10000,8)
#ops = np.random.rand(18)
ops=state
print state
#state=np.reshape(state,(100,1,18))
#ops=np.reshape(ops,(1,18))
inps={"input_1": pos,"input_2":speed,"input_3":state}
type (inps)
mod.fit(inps,ops,batch_size=100,epochs=100000)'''
print('hello')

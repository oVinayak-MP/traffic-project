


import random
import numpy as np
from numpy import newaxis,array
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Concatenate,Input,concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.convolutional import Conv2D,MaxPooling2D


cmd = ['sumo-gui', '-c', '../sumo-map/sumo-map.sumocfg','--waiting-time-memory','7200','-e','500'] #set waiting time meory to maximum time

import os, sys
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
import traci
from tqdm import tqdm


class DQNAgent:
    def __init__(self):

        self.memory = deque(maxlen=8000)
        self.epsilon = 0.6                 #To check exploitive and exploration
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.debug = True
        #self.traci=traci
        self.avgvl=5
        self.lanearrylength=50
        self.maxspeed=15
        self.numlanes=0                    #Not yet used TODO use it inside the neural network
        self.lanelist=['1to5_0','1to5_1','1to5_2','1to5_3','2to5_0','2to5_1','2to5_2','2to5_3','3to5_0','3to5_1','3to5_2','3to5_3','4to5_0','4to5_1','4to5_2','4to5_3']#create a general function for it

        self.actionlist=['GGGGGrrrrrrrrrGGGGrrrrrrGGGG',
                         'rrrrrGGGGGrrrrrrrrrrGGGGGrrr',
                         'rrrrrrrrrrGGGGGrrrrrGGGGGrrr',
                         'GGGGGrrrrrrrrrrGGGGGrrrrrrrr',
                         'rrrrrGGGGGrrrrrrrrrrGGGGGrrr',
                         'rrrrrrrrrrrGGGGGrrrrrrrrrGGG',
                         'rrrrrGGGGGrrrrrrrrrrGGGGGrrr',
                         'GGGGrrrrrrrrrrrrrrrrGGGGGrrr',
                         ]
        self.actionsize=0
        self.setDefaultNumbers()   #To set action size and number of lanes
        self.model=None
        self.batch_size=10
        self.actiontimeperiod=25
        self.actionyellowperiod=7
        return

   

    def printd(self,str):
        if self.debug == True:
            print str
        return

    def add(self,state,reward,nextaction):        #remember

        self.memory.append((state,reward,nextaction))
        return

    def setDefaultNumbers(self):
        self.numlanes=len(self.lanelist)
        self.actionsize=len(self.actionlist)
        return

    def doAction(self,traci,idno,actionindex):
         self.printd("action index is " + str(actionindex))
         traci.trafficlights.setRedYellowGreenState(idno, self.actionlist[actionindex])
         return
    def convertSateto4dim(self,state):
         tempstate={'input_1':state['input_1'][newaxis,:,:,:],'input_2':state['input_2'][newaxis,:,:,:],'input_3':state['input_3'][newaxis,:]}
         print tempstate['input_1'].shape
         print tempstate['input_2'].shape
         print tempstate['input_3'].shape
         return tempstate

    def generateActionIndex(self,state):
         if(np.random.rand()<self.epsilon):
              self.printd("Random action selected")
              return random.randrange(0,self.actionsize,1)
         else :
              if self.debug == True :                      #convert state to format acccepted by predict
                   self.printd("Action predicted ")


              return np.argmax(self.model.predict(self.convertSateto4dim(state)))
         return

    def generateAction(self,state):                              #not yet used
         return actionlist[self.generateActionIndex(state)]

    def generateActionArray(self,ind):

         print "The action size" + str(self.actionsize)
         actionarrt=np.zeros((self.actionsize))
         print "the array shap" + str(actionarrt.shape)
         actionarrt[ind]=1
         return actionarrt

    def learn(self):                                               #could be more efficient
         inp1 =array('i')
         inp2 =array('i')                                       #integer array
         inp3 =array('i')
         targ1=array('i')
         for state,reward,nextstate in self.memory:
               self.printd("the dimes" +str(inp1.shape) + "   "+str(state['input_1'].shape))
               np.append(inp1,state['input_1'],axis=0)
               np.append(inp2,state['input_2'],axis=0)
               np.append(inp3,state['input_3'],axis=0)
               targ=np.zeros(actionsize)
               targ[nextstate]=reward+self.gamma(model.predict(state))
               targ1.append(targ,axis=0)

         inputs={'input_1':inp1,'input_2':inp2,'input_3':inp3}
         model.fit(inputs,targ1,batch_size=self.batch_size,epochs=1,verbose=1)
         return



    def getSpeedandPositionandTime(self,traci,laneid,debug):
        #This function returns speed and position for the give lane it also should contains average vehicle length

        lanepos=[0]*self.lanearrylength
        lanespeed=[0]*self.lanearrylength
        vehct=0
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
            vehct=vehct+traci.vehicle.getAccumulatedWaitingTime(id)
            if (debug >= 1):
                    print "index:" + str(pos)
                    print "speed:" + str(lanespeed[pos])

        if(debug>=1):
             print "number of vehicles:" + str(cnt) +"\n\n"
        lanespeed[:]=[round(x/self.maxspeed,2) for x in lanespeed]
        return lanespeed,lanepos,vehct
    def starttraci(self):
        traci.start(cmd)
        return

    def getStateMatandWaittime(self,traci,lanelist,debug):
        speedl=[]
        posl=[]
        wait=0;
        for x in lanelist:
             speedi,posi,waitt=self.getSpeedandPositionandTime(traci,x,debug-1)
             if (debug>=1) :
                  print "waiting time at " +x + " :" + str(waitt)
             wait=wait+waitt
             speedl.append(speedi)
             posl.append(posi)


        speedl=array(speedl)
        posl=array(posl)

        speedl=speedl[:,:,newaxis]
        posl=posl[:,:,newaxis]
        if (debug >=1) :
             print "waiting time :"+str(wait)
        return speedl,posl,wait


    def create_model(self):                          #creates the model using fumctional API
        first_con_input = Input(shape=(16,50,1))    #size of matrix
        second_con_input = Input(shape=(16,50,1))   #size of matrix
        third_et_input = Input(shape=(8,))         #TODO replace these predefined numbers  to class variables
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
        finalmodel=Dense(8, activation='softmax')(finalemodel_2)     #output row and column TODO replace this with variables
        final=Model(inputs=[first_con_input,second_con_input,third_et_input],outputs=[finalmodel])
        final.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        if self.debug == True:
            print ("The network model")
            final.summary()
        self.model=final
        return final


    def run(self,traci):
         oldw=0
         print self.actionsize
         curstate=random.randrange(0,self.actionsize,1)
         prvstate=curstate
         self.doAction(traci,'5',curstate)
         sp,pos,w=a.getStateMatandWaittime(traci,a.lanelist,1)
         time=1
         while True:
             traci.simulationStep()
             if time%15==0 :
                  state={'input_1':pos,'input_2':sp,'input_3':self.generateActionArray(prvstate)}

                  prvstate=curstate
                  curstate=self.generateActionIndex(state)
                  self.doAction(traci,'5',curstate)
                  sp,pos,w=a.getStateMatandWaittime(traci,a.lanelist,1)
                  reward=w-oldw
                  reward=reward*(-1)                     #negate the reward
                  oldw=w
                  self.add(state,reward,self.generateActionArray(prvstate))
                  print reward
             if time%100==0:
                  self.learn()
             time=time+1




a = DQNAgent()
a.starttraci()
a.setDefaultNumbers()
a.create_model()
a.run(traci)
#a.getSpeedandPosition(traci,)
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

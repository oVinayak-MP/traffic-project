

from sklearn.metrics import mean_squared_error
import random
import numpy as np
from numpy import newaxis,array
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Concatenate,Input,concatenate
from keras.optimizers import Adam,RMSprop
from keras import backend as K
from keras.layers.convolutional import Conv2D,MaxPooling2D
import matplotlib.pyplot as plt

cmd = ['sumo-gui', '-c', '../sumo-map/sumo-map.sumocfg','--waiting-time-memory','999999','-e','500','--time-to-teleport','-5'] #set waiting time meory    maximum time change gui mode

import os, sys
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
import traci
from tqdm import tqdm


class DQNAgent:
    def __init__(self):

        self.memory = deque(maxlen=18000)     #max size of queue
        self.epsilon = 0.6                 #To check exploitive and exploration
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.debug = True
        self.gamma = 0.05
        #self.traci=traci
        self.avgvl=5
        self.lanearrylength=50
        self.maxspeed=15
        self.numlanes=0                    #Not yet used TODO use it inside the neural network
        self.lanelist=['1to5_0','1to5_1','1to5_2','1to5_3','2to5_0','2to5_1','2to5_2','2to5_3','3to5_0','3to5_1','3to5_2','3to5_3','4to5_0','4to5_1','4to5_2','4to5_3']#create a general function for it

        self.actionlist=['GGrrrrrrrrrrrrrrrrrrrrrrrrrr',
                         'rrGGrrrrrrrrrrrrrrrrrrrrrrrr',
                         'rrrrGGrrrrrrrrrrrrrrrrrrrrrr',
                         'rrrrrrGGrrrrrrrrrrrrrrrrrrrr',
                         'rrrrrrrrGGrrrrrrrrrrrrrrrrrr',
                         'rrrrrrrrrrGGrrrrrrrrrrrrrrrr',
                         'rrrrrrrrrrrrGGrrrrrrrrrrrrrr',
                         'rrrrrrrrrrrrrrGGrrrrrrrrrrrr',
                         'rrrrrrrrrrrrrrrrGGrrrrrrrrrr',
                         'rrrrrrrrrrrrrrrrrrGGrrrrrrrr',
                         'rrrrrrrrrrrrrrrrrrrrGGrrrrrr',
                         'rrrrrrrrrrrrrrrrrrrrrrGGrrrr',
                         'rrrrrrrrrrrrrrrrrrrrrrrrGGGG'

                         ]
        self.actionsize=0
        self.time=0
        self.history=0
        self.setDefaultNumbers()   #To set action size and number of lanes
        self.model=None
        self.batch_size=32
        self.actiontimeperiod=25
        self.actionyellowperiod=7
        self.epoch=8000
        self.mseplot=[]
        return

    def loadmodelweights(self,filename):
        self.model.load_weights(filename)
        return
    def savemodelweights(self,filename):
        self.model.save_weights(filename)
        return
    def loadmodel(self,filename):
        self.model.load(filename)
        return
    def savemodel(self,filename):
        self.model.save(filename)
        return
   

    def printd(self,str):
        if self.debug == True:
            print str
        return

    def add(self,state,reward,nextaction,nextstate):        #remember

        self.memory.append((state,reward,nextaction,nextstate))
        return

    def setDefaultNumbers(self):
        self.numlanes=len(self.lanelist)
        self.actionsize=len(self.actionlist)
        return

    def doAction(self,traci,idno,actionindex):
         self.printd("action index is " + str(actionindex))
         traci.trafficlights.setRedYellowGreenState(idno, self.actionlist[actionindex])
         return
    def doActionStr(self,traci,idno,strs):
         traci.trafficlights.setRedYellowGreenState(idno,strs)

    def convertSateto4dim(self,state):                                        #converts it into the state accpeted by the predict function
         tempstate={'input_1':state['input_1'][newaxis,:,:,:],'input_2':state['input_2'][newaxis,:,:,:],'input_3':state['input_3'][newaxis,:]}

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


         actionarrt=np.zeros((self.actionsize),dtype=float)

         actionarrt[ind]=1
         return actionarrt
    def learndummy(self):
         while True:
              inp1=np.zeros((500,16,50,1))
              inp2=np.zeros((500,16,50,1))
              inp3=np.zeros((500,self.actionsize))
              inp3[:][0]=1
              targ=np.zeros((500,self.actionsize))
              targ[:][0]=10000


              inputs={'input_1':inp1,'input_2':inp2,'input_3':inp3}
              print "inputs"+str(inputs)
              print "target"+str(targ)


              self.model.fit(inputs,targ,batch_size=32,epochs=1,verbose=1)
              print self.model.predict(inputs)
              input()




    def learn(self):                                               #could be more efficient
         inp1 =[]
         inp2 =[]
         inp3 =[]
         targ1=[]
         realrewards=[]
         calcrewards=[]
         for state,reward,action,nextstate in self.memory:

               inp1.append(state['input_1'])
               inp2.append(state['input_2'])
               inp3.append(state['input_3'])
               targ=np.zeros(self.actionsize)


               temparry=self.model.predict(self.convertSateto4dim(state))



               print temparry[0]
               temp=np.argmax(temparry[0])
               nextstate['input_3']=self.generateActionArray(temp)
               nextreward=self.model.predict(self.convertSateto4dim(nextstate))       #predict thee next reward
               nextreward=nextreward[0]
               nextreward=np.max(nextreward)
               print "current action " + str(action)
               print "selected action" + str(temp)
               temp=temparry[0][temp]
               targ[action]=reward +self.gamma*nextreward
               realrewards.append(reward)
               calcrewards.append(temparry[0][action])

               print "original reward is " + str(reward) + "and calculated reward is" + str(targ[action])
               print targ
               targ1.append(targ)

         inp1=np.array(inp1)
         inp2=np.array(inp2)
         inp3=np.array(inp3)
         targ1=np.array(targ1)

         inputs={'input_1':inp1,'input_2':inp2,'input_3':inp3}
         self.history=self.model.fit(inputs,targ1,batch_size=self.batch_size,epochs=2,verbose=1)
         mse=mean_squared_error(realrewards,calcrewards)
         self.mseplot.append(mse)
         return
    def generateIntermediateAction(self,action1,action2):
         lens=len(action1)
         temp=action1
         for i in range(0,lens):
              if action1[i]!=action2[i]:
                   if action1[i]=='G' or action1[i]=='g' :
                        temp=temp[:i] + 'Y' + temp[i+1:]                #remove condition if you want yellow between red to green transition

         return temp



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
        first_con_input = Input(shape=(self.numlanes ,self.lanearrylength,1))    #size of matrix
        second_con_input = Input(shape=(self.numlanes ,self.lanearrylength,1))   #size of matrix
        third_et_input = Input(shape=(self.actionsize,))         #TODO replace these predefined numbers  to class variables
        model1_1=Conv2D(16, kernel_size=(4, 4), strides=(2, 2), activation='relu',data_format='channels_last',padding='same')(first_con_input)
        model2_1=Conv2D(16, kernel_size=(4, 4), strides=(2, 2), activation='relu',data_format='channels_last',padding='same')(second_con_input)
        model1_2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model1_1)	#
        model2_2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model2_1)
        model1_3=Conv2D(32, (2, 2), activation='relu')(model1_2)
        model2_3=Conv2D(32, (2, 2), activation='relu')(model2_2)
        model1_4=MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(model1_3)	#
        model2_4=MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(model2_3)	#
        model1_5=Flatten()(model1_4)
        model2_5=Flatten()(model2_4)
        #model1_6=Dense(256, activation='relu')(model1_5)                 #might not need these
        #model2_6=Dense(256, activation='relu')(model2_5)
        #model3_1=Dense(self.actionsize, activation='relu')(third_et_input) #size of state
        #model3_2=Flatten()(model3_1)  #not used
        tempmodel=concatenate([model1_5,model2_5],axis=-1)
        tempmodel.shape
        finalemodel_1=concatenate([tempmodel,third_et_input],axis=-1)
        finalemodel_2=Dense(128, activation='relu')(finalemodel_1)      #change values
        finalemodel_3=Dense(64, activation='relu')(finalemodel_2)      #change values
        finalmodel=Dense(self.actionsize, activation='linear')(finalemodel_3)     #output row and column TODO replace this with variables
        final=Model(inputs=[first_con_input,second_con_input,third_et_input],outputs=[finalmodel])
        final.compile(loss='mean_squared_error',optimizer=RMSprop(lr=0.006),metrics=['accuracy'])
        if self.debug == True:
            print ("The network model")
            final.summary()
        self.model=final
        return final

    def threadexplorer(self,a):
         self.starttraci()



    def run(self,traci):
         oldw=0
         print self.actionsize
         curstate=random.randrange(0,self.actionsize,1)
         prvstate=curstate
         self.doAction(traci,'5',curstate)
         sp,pos,w=a.getStateMatandWaittime(traci,a.lanelist,1)
         self.time=4
         self.epsilon=1.0
         plotx=[]
         wt=[]
         teleportime =[]
         tt=0
         while self.time<720000:
             traci.simulationStep()
             if self.time%12==0 :
                  state={'input_1':pos,'input_2':sp,'input_3':self.generateActionArray(prvstate)}
                  #self.printd("doing yellow")
                  prvstate=curstate
                  curstate=self.generateActionIndex(state)
                  sp,pos,w=a.getStateMatandWaittime(traci,a.lanelist,1)
                  nextstate={'input_1':pos,'input_2':sp,'input_3':0}
                  reward=w-oldw
                  reward=reward*(-1)                     #negate the reward
                  oldw=w
                  plotx.append(reward)
                  wt.append(w)
                  self.add(state,reward,prvstate,nextstate)                                         #add to the buffer the previous state its reward and action taken at that state
                  yellowaction=a.generateIntermediateAction(self.actionlist[prvstate],self.actionlist[curstate])
                  #self.printd("yellow action is"+str(yellowaction))
                  self.doActionStr(traci,'5',yellowaction)                                                #sets yellow state and calculates change cumulative delay for previous action


                  print "reward is " + str(reward)
             if self.time%12==3 :
                  #self.printd("doing action")
                  self.doAction(traci,'5',curstate)                                                #sets the current state after the yellow transition

             if self.time%80000==0:
                  self.learn()
                  self.epsilon=self.epsilon-0.01
                  print "epislion" +str(self.epsilon)
             if self.time%80000==0:
                  fg,(pl1,pl2)=plt.subplots(2,1)
                  pl1.plot(plotx)
                  pl1.plot(wt)
                  pl2.plot(self.mseplot)
                  print self.mseplot
                  self.savemodelweights("mod.wt")
                  print(self.history.history)
                  plt.show()
             self.time=self.time+1
             tt=tt+traci.simulation.getEndingTeleportNumber()                              #TODO also consider members teleported
             teleportime.append(traci.simulation.getEndingTeleportNumber())




np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
a = DQNAgent()

a.starttraci()
a.setDefaultNumbers()
a.create_model()
if (os.path.exists('mdel.h5') and False ):                       #change this to load model
    a.loadmodel("mdel.h5")
elif(os.path.exists('mod.wt')):
    a.loadmodelweights("mod.wt")
a.run(traci)
#a.learndummy()
a.savemodel('mdel.h5')
a.savemodelweights("mod.wt")

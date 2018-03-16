


import random
import numpy as np
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Concatenate,Input,concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.convolutional import Conv2D,MaxPooling2D

class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size=state_size
        self.action_size=action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.00
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.debug = True

    def printd(self,str):
        if self.debug == True:
            print str

    def convnet():              #creates the required convolutional network
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid',input_shape=(12,200,1),data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))	#
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

    def merger_and_net(con1,con2): #merges convnet and also takes input state
        model=Sequential()
        model.add(Dense(18, input_shape=(1,1,18), activation='sigmoid')) #size of state
        merged = Concatenate([con1, con2])
        result = Concatenate([merged,model])
        result = Dense(530, activation='relu')

        #results = Model(inputs=[con1,con2,model],result)
        result.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return results

    def create_model(self):                          #creates the model using fumctional API
        first_con_input = Input(shape=(12,20,1))    #size of matrix
        second_con_input = Input(shape=(12,20,1))   #size of matrix
        third_et_input = Input(shape=(18,))
        model1_1=Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid',input_shape=(12,200,1),data_format='channels_last',padding='same')(first_con_input)
        model2_1=Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid',input_shape=(12,200,1),data_format='channels_last',padding='same')(second_con_input)
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
        finalmodel=Dense(18, activation='softmax')(finalemodel_2) #output row and column
        final=Model(inputs=[first_con_input,second_con_input,third_et_input],outputs=[finalmodel])
        final.compile(loss='categorical_crossentropy',optimizer=Adam(lr=self.learning_rate),metrics=['accuracy'])
        if self.debug == True:
            print ("The network model")
            final.summary()
        return final
a = DQNAgent(100,200)
mod=a.create_model()
pos= np.random.rand(10000,12,20,1)
#pos= np.random.rand(12,20)
#pos=np.reshape(pos,(1,12,20,1))
#speed= np.random.rand(12,20)
#speed= np.reshape(speed,(1,12,20,1))
speed= np.random.rand(10000,12,20,1)
state = np.random.rand(10000,18)
#ops = np.random.rand(18)
ops=state
print state
#state=np.reshape(state,(100,1,18))
#ops=np.reshape(ops,(1,18))
inps={"input_1": pos,"input_2":speed,"input_3":state}
type (inps)
mod.fit(inps,ops,batch_size=100,epochs=100000)
#print pos
print('hello')


import random
import numpy as np
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Concatenate,Input,concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.convolutional import Conv2D,MaxPooling2D

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
    model.add(Dense(18, input_shape=(1,18,1), activation='sigmoid')) #size of state
    merged = Concatenate([con1, con2])
    result = Concatenate([merged,model])
    result = Dense(530, activation='relu')

    #results = Model(inputs=[con1,con2,model],result)
    result.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return results

def create_model():
    first_con_input = Input(shape=(12,200,1))
    second_con_input = Input(shape=(12,200,1))
    third_et_input = Input(shape=(1,18))
    model1_1=Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid',input_shape=(12,200,1),data_format='channels_last')(first_con_input)
    model2_1=Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='sigmoid',input_shape=(12,200,1),data_format='channels_last')(second_con_input)
    model1_2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model1_1)	#
    model2_2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model2_1)
    model1_3=Conv2D(32, (2, 2), activation='relu')(model1_2)
    model2_3=Conv2D(32, (2, 2), activation='relu')(model2_2)
    model1_4=MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(model1_3)	#
    model2_4=MaxPooling2D(pool_size=(1, 2), strides=(2, 2))(model2_3)	#
    model1_5=Flatten()(model1_4)
    model2_5=Flatten()(model2_4)
    model1_6=Dense(256, activation='relu')(model1_5)
    model2_6=Dense(256, activation='relu')(model1_6)
    model3_1=Dense(18, input_shape=(1,18,1), activation='sigmoid')(third_et_input) #size of state
    model3_1=Flatten()(model3_1)
    tempmodel=concatenate([model1_6,model2_6])
    finalemodel_1=concatenate([tempmodel,model3_1],axis=1)
    finalemodel_2=Dense(530, activation='relu')(finalemodel_1)

    final=Model(inputs=[first_con_input,second_con_input,third_et_input],outputs=[finalemodel_2])
    final.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print (final.summary())
    return final
create_model()

print('hello')

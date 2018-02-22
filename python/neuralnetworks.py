
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Flatten,Concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.convolutional import Conv2D,MaxPooling2D

def convnet():              #creates the required convolutional network
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 2), activation='sigmoid',input_shape=(12,200,1),data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))	#
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    return model

def merger_and_net(con1,con2): #merges convnet and also takes input state
    model=Sequential()
    model.add(Dense(18, input_shape=(1,18,1), activation='sigmoid')) #size of state
    merged = Concatenate([con1, con2])
    result = Concatenate([merged,model])
    result=Dense(530, activation='relu')
    result= Dense(128, activation='relu')
    return result
a=convnet()
b=convnet()
merger_and_net(a,b)

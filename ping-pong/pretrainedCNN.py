# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# and uses Keras.
import os
import gym
import cv2
import argparse
import sys, glob
import numpy as np
import _pickle as pickle
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling2D, Convolution2D

from keras.layers import Flatten, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16
import glob
import cv2
import numpy as np

import scipy.io as sio

from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM, Flatten, Conv1D, Dropout,BatchNormalization,GRU,MaxPooling1D,LeakyReLU,GlobalAveragePooling1D
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#Script Parameters
input_dim = 80*80
gamma = 0.99
update_frequency = 1
learning_rate = 0.001
resume = True
render = True

#Initialize
env = gym.make("Pong-v0")
number_of_inputs = env.action_space.n #This is incorrect for Pong (?)
#number_of_inputs = 1
observation = env.reset()
prev_x = None
xs, dlogps, drs, probs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
train_X = []
train_y = []

#def pong_preprocess_screen(I):
#  I = I[35:195] 
#  I = I[::2,::2,0] 
#  I[I == 144] = 0 
#  I[I == 109] = 0 
#  I[I != 0] = 1 
#  return I.astype(np.float).ravel()

#Define the main model (WIP)
def model_pre(input_dim=80*80, model_type=1):
  model = Sequential()
  if model_type==0:
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Flatten())
    model.add(Dense(200, activation = 'relu'))
    model.add(Dense(number_of_inputs, activation='softmax'))
  else:
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform',name='output'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu', init='he_uniform',name='output2'))
    model.add(Dense(number_of_inputs, activation='softmax'))
  return model
model_pre=model_pre()
model_pre.load_weights('pong_checkpoint.h5')
model_pre.summary()
dense1_layer_model = Model(inputs=model_pre.input,
                           outputs=model_pre.get_layer('output').output)
def pong_preprocess_screen(I,dense1_layer_model):
  I = I[35:195] 
  I = I[::2,::2,0] 
  I[I == 144] = 0 
  I[I == 109] = 0 
  I[I != 0] = 1
  I=I.astype(np.float).ravel()
  #print(I.shape)
  #input = Input(shape=(80,80), name = 'image_input')
  #our_feats = vgg_model(input)
  #our_feats = Flatten()(our_feats)
  

  #model = Model(inputs=input, outputs=our_feats)
  #dense1_layer_model.summary()
  output_feats = dense1_layer_model.predict(I.reshape([1,I.shape[0]])) # Extract features of our images
  
  output_feats=output_feats.astype(np.float).ravel()
  return output_feats
def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

#Define the main model (WIP)
input_dim2=640
def learning_model(input_dim2=640, model_type=1):
  model = Sequential()
  if model_type==0:
    model.add(Reshape((1,640), input_shape=(input_dim2,)))
    model.add(Flatten())
    model.add(Dense(200, activation = 'relu'))
    model.add(Dense(number_of_inputs, activation='softmax'))
    opt = RMSprop(lr=learning_rate)
  else:
    model.add(Reshape((1,640), input_shape=(input_dim2,)))
    model.add(Conv1D(32,1, input_shape=(1,640), padding="same", strides=10))
    model.add(Flatten())
    model.add(Dense(16, activation='relu', init='he_uniform'))
    model.add(Dense(number_of_inputs, activation='softmax'))
    opt = Adam(lr=learning_rate)
  model.compile(loss='categorical_crossentropy', optimizer=opt)
  if resume == True:
    model.load_weights('pong_model_checkpoint.h5')
  return model

model = learning_model()
running_reward_label=[]
reward_sum_label=[]


#Begin training
while True:
  if render: 
    env.render()
  #Preprocess, consider the frame difference as features
  cur_x = pong_preprocess_screen(observation,dense1_layer_model)
  x = cur_x - prev_x if prev_x is not None else np.zeros(640)
  prev_x = cur_x
  #Predict probabilities from the Keras model
  aprob = ((model.predict(x.reshape([1,x.shape[0]]), batch_size=1).flatten()))
  
  '''
  dense1_layer_model = Model(inputs=vgg_model.input,
                                         outputs=vgg_model.get_layer('output2').output)
  #model = Model(inputs=input, outputs=our_feats)
  output_feats = dense1_layer_model.predict(x.reshape([1,x.shape[0]])) # Extract features of our images
  #print(output_feats.shape)
  '''
  
  #aprob = aprob/np.sum(aprob)
  #aprob = aprob/np.sum(aprob)
  #Sample action
  #action = np.random.choice(number_of_inputs, 1, p=aprob)
  #Append features and labels for the episode-batch
  xs.append(x)
  probs.append((model.predict(x.reshape([1,x.shape[0]]), batch_size=1).flatten()))
  aprob = aprob/np.sum(aprob)
  action = np.random.choice(number_of_inputs, 1, p=aprob)[0]
  y = np.zeros([number_of_inputs])
  y[action] = 1
  #print action
  dlogps.append(np.array(y).astype('float32') - aprob)
  observation, reward, done, info = env.step(action)
  reward_sum += reward
  drs.append(reward) 
  if done:
    episode_number += 1
    epx = np.vstack(xs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    epdlogp *= discounted_epr
    #Slowly prepare the training batch
    train_X.append(xs) 
    train_y.append(epdlogp)
    xs,dlogps,drs = [],[],[]
    #Periodically update the model
    if episode_number % update_frequency == 0: 
      y_train = probs + learning_rate * np.squeeze(np.vstack(train_y)) #Hacky WIP
      #y_train[y_train<0] = 0
      #y_train[y_train>1] = 1
      #y_train = y_train / np.sum(np.abs(y_train), axis=1, keepdims=True)
      print('Training Snapshot:')
      print(y_train)
      model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)
      #Clear the batch
      train_X = []
      train_y = []
      probs = []
      #Save a checkpoint of the model
      os.remove('pong_model_checkpoint.h5') if os.path.exists('pong_model_checkpoint.h5') else None
      model.save_weights('pong_model_checkpoint.h5')
    #Reset the current environment nad print the current results
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    reward_sum_label.append(reward_sum)
    running_reward_label.append(running_reward)
    sio.savemat('./reward_sum.mat',{"reward_sum":reward_sum_label,"running_reward":running_reward_label})
    print('Environment reset imminent. Total Episode Reward: %f. Running Mean: %f' % (reward_sum, running_reward))
    reward_sum = 0
    observation = env.reset()
    prev_x = None
  if reward != 0:
    print ('Episode %d Result: ' % episode_number + 'Defeat!' if reward == -1 else 'VICTORY!')

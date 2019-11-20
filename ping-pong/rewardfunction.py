# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# and uses Keras.
import os
import gym
import cv2
import argparse
import sys, glob
import numpy as np
#import cPickle as pickle
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
import scipy.io as sio
#Script Parameters
input_dim = 80 * 80
gamma = 0.99
update_frequency = 1
learning_rate = 0.001
resume = True
render = False

#Initialize
env = gym.make("Pong-v0")
number_of_inputs = env.action_space.n #This is incorrect for Pong (?)
#number_of_inputs = 1
observation = env.reset()
prev_x = None
xs, dlogps, drs, probs = [],[],[],[]
running_reward = None
running_reward2 = None
reward_sum = 0
reward_sum2 = 0
episode_number = 0
train_X = []
train_y = []
running_reward_label=[]
reward_sum_label=[]
def pong_preprocess_screen(I):
  I = I[35:195] 
  I = I[::2,::2,0] 
  I[I == 144] = 0 
  I[I == 109] = 0 
  I[I != 0] = 1 
  return I.astype(np.float).ravel()
def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def patpat(I,reward):
  I = I[35:195]
  I = I[::2,80:,:]
  #print(I) 
  I1 = I[:,:,0]
  I1[I1 == 144] = 0 
  I1[I1 == 109] = 0
  I1[I1 == 92] = 1
  I1[I1 == 236] = 2
  #print(np.where(I1==1))
  #print(np.where(I1==2))
  #print(I1[I1 != 0]) 
  #I1[I1 != 0] = 1
  #I2= I[:,:,1]
  #I3=I[:,:,2]
  pong=(I1==2)
  #print(np.where(I1==1))
  pat=(I1==1)
  #print(np.any(pong[:,59:64]==True))
  patpat=0
  if reward==0:
    if np.any(I1[:,59:64]==2):
      #print(np.where(I1[:,59:64]==2)[0][0])
      #print(np.where(I1[:,59:64]==1)[0])
      if np.any(np.where(I1[:,59:64]==2)[0][0]==np.where(I1[:,59:64]==1)[0]):
        patpat=1
        #print("yesyes")
  else:
    if reward==1:
      patpat=2
    if reward==-1:
      patpat=-1
  return float(patpat)



#Define the main model (WIP)
def learning_model(input_dim=80*80, model_type=1):
  model = Sequential()
  if model_type==0:
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Flatten())
    model.add(Dense(200, activation = 'relu'))
    model.add(Dense(number_of_inputs, activation='softmax'))
    opt = RMSprop(lr=learning_rate)
  else:
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu', init='he_uniform'))
    model.add(Dense(number_of_inputs, activation='softmax'))
    opt = Adam(lr=learning_rate)
  model.compile(loss='categorical_crossentropy', optimizer=opt)
  if resume == True:
      if os.path.exists('pong_model_checkpoint.h5'):
          model.load_weights('pong_model_checkpoint.h5')

  return model

model = learning_model()
if os.path.exists('./reward_sum.mat'):
    if resume == True:
        data=sio.loadmat("./reward_sum.mat")
        running_reward_label=data["running_reward"][0]
        reward_sum_label=data["reward_sum"][0]
        episode_number=len(reward_sum_label)
        reward_sum_label=reward_sum_label.tolist()
        running_reward_label=running_reward_label.tolist()
#Begin training

while True:
  if render: 
    env.render()
  #Preprocess, consider the frame difference as features
  cur_x = pong_preprocess_screen(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
  prev_x = cur_x
  #Predict probabilities from the Keras model
  aprob = ((model.predict(x.reshape([1,x.shape[0]]), batch_size=1).flatten()))
  #print(aprob)
  #aprob = aprob/np.sum(aprob)
  #Sample action
  #action = np.random.choice(number_of_inputs, 1, p=aprob)
  #Append features and labels for the episode-batch
  xs.append(x)
  probs.append((model.predict(x.reshape([1,x.shape[0]]), batch_size=1).flatten()))
  aprob = aprob/np.sum(aprob)
  #print(aprob)
  action = np.random.choice(number_of_inputs, 1, p=aprob)[0]
  y = np.zeros([number_of_inputs])
  y[action] = 1
  #print action
  dlogps.append(np.array(y).astype('float32') - aprob)
  observation, reward, done, info = env.step(action)
  #print(reward)
  reward_pat=patpat(observation,reward)
  #print(reward_pat)
  reward_sum2 +=reward
  reward_sum += reward_pat
  drs.append(reward_pat) 
  if done:
    
    episode_number += 1
    epx = np.vstack(xs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    discounted_epr = discount_rewards(epr)
    #print(discounted_epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    #print(discounted_epr)
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
      #print('Training Snapshot:')
      #print (y_train)
      model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)
      #Clear the batch
      train_X = []
      train_y = []
      probs = []
      #Save a checkpoint of the model
      os.remove('pong_checkpoint2.h5') if os.path.exists('pong_checkpoint2.h5') else None
      model.save_weights('pong_checkpoint2.h5')
    #Reset the current environment nad print the current results
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    running_reward2 = reward_sum2 if running_reward2 is None else running_reward2 * 0.99 + reward_sum2 * 0.01
    print ('Environment reset imminent. Episode %f. Total Episode Reward: %f. Running Mean: %f' % (episode_number,reward_sum, running_reward))
    print ('Environment reset imminent. Episode %f. Total Episode Reward2: %f. Running Mean2: %f' % (episode_number,reward_sum2, running_reward2))    
    reward_sum_label.append(reward_sum2)
    #print(reward_sum_label)
    running_reward_label.append(running_reward2)
    sio.savemat('./reward_sum.mat',{"reward_sum":reward_sum_label,"running_reward":running_reward_label})
   
    reward_sum = 0
    reward_sum2 = 0
    observation = env.reset()
    prev_x = None
  #if reward != 0:
   # print (('Episode %d Result: ' % episode_number) + ('Defeat!' if reward == -1 else 'VICTORY!'))
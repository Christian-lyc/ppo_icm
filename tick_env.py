import copy
from typing import Optional, Tuple
import random
import gym
import torch
from gym import spaces
import numpy as np
import os
import cv2
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import namedtuple
#from dataset import Dataset
#from DQN_tick import DQN

from gym.core import ObsType, ActType

Rectangle = namedtuple(
    'Rectangle', [
        'xmin', 'xmax', 'ymin', 'ymax'])

class TickEnv(gym.Env):
    def __init__(self, dataset,args,image_dim=None,step_size=5,agents=1):
        super(TickEnv,self).__init__()
        self.agents=agents
        self.image_size=[image_dim[0],image_dim[1]]

        self.train_loader_iter=dataset.sample_data()#iter(self.train_loader)
        self.test_loader_iter=dataset.test_sample_data()

        self.step_size = step_size
        self.action_space = spaces.Discrete(2)
        self.actions = self.action_space.n
        self.observation_space = spaces.Box(low=0, high=10, shape=[7,7],
                                                 dtype=np.float32)  # shape change to output shape
        self.task=args.task
        #self._restart_episode()


    def reset(self,task):
        self._restart_episode(task)
        return self._current_state()
    def _restart_episode(self,task):
        self.reward=np.zeros((self.agents,))
        self.terminal=[False]*self.agents
        #next -->next batch
        #constantly load the image without end
        self.task=task
        if self.task=='train':
            self._image,self._label = next(self.train_loader_iter)
        elif self.task=='test':
            self._image,self._label=next(self.test_loader_iter)
#        x_min=np.array([5])
#        x_max=np.array([5])
#        y_min=np.array([5])
#        y_max=np.array([5])
        ### left right up down initial state of image croping
#        if self.task=='train':
#            while x_min%2!=0:
#                x_min = np.random.randint(
#                    0,15,
#                    self.agents)
#            while x_max%2!=0:
#                x_max=np.random.randint(
#                    self.image_size[0]-15,
#                    self.image_size[0],
#                    self.agents)
#            while y_min%2!=0:
#                y_min=np.random.randint(
#                    0,15,
#                    self.agents)
#            while y_max%2!=0:
#                y_max=np.random.randint(
#                    self.image_size[1]-15, self.image_size[1],
#                    self.agents)
#        else:
        x_min=np.array([10])
        x_max=np.array(self.image_size[0])-np.array([10])
        y_min=np.array([10])
        y_max=np.array(self.image_size[1])-np.array([10])
        self._location=[(x_min,x_max,y_min,y_max) for i in range(self.agents)]
        #self._crop_image=self._image.data[x_min:x_max,y_min:y_max]
        #self._qvalues = [[0, ] * self.actions] * self.agents

    def _current_state(self):

        return self._image


    def step(self, act,class_pred,step,update_image):

        self.terminal=[False]*self.agents
        #current_loc=self._location

        #next_location=copy.deepcopy(current_loc)
        #act from output of network (after softmax) 1*6 one hot
        for i in range(self.agents):
            if self.task=='train':
                class_reward = class_pred == self._label[0]
                if self._label[0]==0:
                    reward_lambda=42/115
                elif self._label[0]==1:
                    reward_lambda=42/43
                elif self._label[0]==2:
                    reward_lambda=42/42
                elif self._label[0]==3:
                    reward_lambda=42/1576
                elif self._label[0]==4:
                    reward_lambda=42/2711
                elif self._label[0]==5:
                    reward_lambda=42/1169
                elif self._label[0]==6:
                    reward_lambda=42/47
            if step==0 and act==0:
                self._crop_image = update_image
                self.reward[i] =-0.1
            elif step==0 and act==1:
                self._crop_image = update_image[:,:,self._location[0][0][0]:, :]#left
                self.reward[i] =-0.1
            elif step==0 and act==2:
                if self.task=='train':
                    if class_reward:
                        self.reward[i]=+1*reward_lambda-0.1
                        self.terminal = [True] * self.agents
                    else:
                        self.reward[i]=-1*reward_lambda-0.1
                    self._crop_image = update_image
                elif self.task=='test':
                    self.terminal = [True] * self.agents

            if step==1 and act==0:
                self._crop_image=update_image
                self.reward[i] = -0.1
            elif step==1 and act==1:
                self._crop_image=update_image[:,:,:self._location[0][1][0],:]#right
                self.reward[i] = -0.1
            elif step==1 and act==2:
                if self.task=='train':
                    if class_reward:
                        self.reward[i]=+1*reward_lambda-0.1
                        self.terminal = [True] * self.agents
                    else:
                        self.reward[i]=-1*reward_lambda-0.1
                    self._crop_image = update_image
                elif self.task=='test':
                    self.terminal = [True] * self.agents

            if step==2 and act==0:
                self._crop_image=update_image
                self.reward[i] = -0.1
            elif step==2 and act==1:
                self._crop_image=update_image[:,:,:,self._location[0][2][0]:]#down
                self.reward[i] = -0.1
            elif step==2 and act==2:
                if self.task=='train':
                    if class_reward:
                        self.reward[i]=+1*reward_lambda-0.1
                        self.terminal = [True] * self.agents
                    else:
                        self.reward[i]=-1*reward_lambda-0.1
                    self._crop_image = update_image
                elif self.task=='test':
                    self.terminal = [True] * self.agents

            if step==3 and act==0:
                self._crop_image=update_image
                self.reward[i] = -0.1
            elif step==3 and act==1:
                self._crop_image=update_image[:,:,:,:self._location[0][3][0]]#upself.reward[i] =-0.1
                self.reward[i] = -0.1
            elif step==3 and act==2:
                if self.task=='train':
                    if class_reward:
                        self.reward[i]=+1*reward_lambda-0.1
                        self.terminal = [True] * self.agents
                    else:
                        self.reward[i]=-1*reward_lambda-0.1
                    self._crop_image = update_image
                elif self.task=='test':
                    self.terminal = [True] * self.agents

            if class_pred==self._label[0] and act!=2:
                self.reward[i]=self.reward[i]-1
             ###

        info={}

        return self._crop_image,self.reward.copy(), self.terminal,info













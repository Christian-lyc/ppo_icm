import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from collections import deque
import random
import resnet




class PPO_tick(nn.Module):
    def __init__(self,observation:int,num_class:int):
        super(PPO_tick,self).__init__()
        self.linear1=nn.Linear(observation,1024)
        self.linear2=nn.Linear(1024,512)
        self.actor_layer=nn.Linear(512,num_class)
        self.critic_layer=nn.Linear(512,1)
        self.relu=nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)



    def forward(self,x,location):
        x=self.avgpool(x)
        x = torch.cat((x.squeeze(-1).squeeze(-1), location),dim=-1)
        bs,feature_size=x.view((x.shape[0],x.shape[1])).shape
        x=self.linear1(x.squeeze(-1).squeeze(-1))
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        output=self.actor_layer(x)
        output=F.softmax(output,dim=1)
        value=self.critic_layer(x)
        #output=output.reshpe(bs,4,-1)
        return output,value 
    


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class ICM(nn.Module):
    def __init__(self,num_inputs,num_actions):
        super(ICM,self).__init__()
        self.inverse_net=nn.Sequential(
            nn.Linear(num_inputs*2,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,num_actions)
        )
        self.forward_net=nn.Sequential(
            nn.Linear(num_inputs+num_actions,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,num_inputs)
        )
        self.avgpool = nn.AvgPool2d(7)

    def forward(self,state,next_state,action,locations,next_locations):
        state = self.avgpool(state).squeeze(2).squeeze(2)
        state=torch.cat((state,locations),dim=1)
        next_state = self.avgpool(next_state).squeeze(2).squeeze(2)
        next_state=torch.cat((next_state,next_locations),dim=1)
        action_pred=self.inverse_net(torch.cat((state,next_state),dim=1))
        next_state_pred=self.forward_net(torch.cat((state,action),dim=1))
        return action_pred,next_state_pred


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class ReplayMemory(object):
    def __init__(self,max_size,state_shape,agents=1):
        self.max_size=int(max_size)
        self.state_shape=state_shape
        self.agents=agents
        #self.state=np.zeros((self.agents, self.max_size) + state_shape, dtype='uint8')
        #self.action=np.zeros((self.agents,self.max_size), dtype='int32')
        #self.reward = np.zeros((self.agents, self.max_size), dtype='float32')
        #self.isOver = np.zeros((self.agents, self.max_size), dtype='bool')

        #set up a plug in and pop out queue
        self.memory=deque([],maxlen=max_size)

    def append_transition(self,*args):
        self.memory.append(Transition(*zip(*args)))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

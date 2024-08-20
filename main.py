import numpy as np
from PPO_tick import PPO_tick,ICM
from tick_env import TickEnv
import torch
import torch.nn.functional as F
from test import Test
import torch.nn as nn
import argparse
import os
from dataset import Dataset

from torch.distributions.categorical import Categorical
import resnet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
class Training(object):
    def __init__(self,dataset,env,image_size,args,obs_ch=2048,agents=1,action_step=4,number_action_seq=4,
                num_class=21,gamma=0.99):
        super(Training,self).__init__()

        self.env=env
        self.image_size=(image_size,image_size)
        self.gamma=gamma
        self.steps_per_episode=args.steps_per_episode
        self.number_action_seq=number_action_seq
        self.max_episodes=args.epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.pretrain_network = resnet.resnet50(pretrained=True).train(False).to(self.device)
        self.action_step=action_step
        self.agents=agents
        self.num_class=num_class
        self.a_c_network = PPO_tick(obs_ch+4, num_class).to(self.device) #tanh critic
        self.test=Test(env,image_size,self.pretrain_network,self.a_c_network,args,agents=1,max_steps=len(dataset.test_loader))#dataset.test_loader
        self.task=args.task
        self.lr=args.lr
        self.icm=ICM(obs_ch+4,num_class).to(self.device)
        self.inv_criterion = nn.CrossEntropyLoss()
        self.fwd_criterion = nn.MSELoss()
        self.avgpool = nn.AvgPool2d(7)
        
        self.optimizer = torch.optim.Adam(list(self.a_c_network.parameters())+list(self.icm.parameters()), lr=args.lr,eps=1e-05)


    def train(self):
        episode=1

        epoch_returns = []
        best_prec1=0
        lowest_loss = 10
        lowest=False

        train_log=[]
        explore_label=[]

        while episode<=self.max_episodes:
            terminal = [False] * self.agents
            states = []
            actions = []
            rewards_tol = []
            log_prob_action_k = []
            values = []
            next_states=[] ##
            termination = []
            break_out_flag = False
            obs=self.env.reset(self.task)
            location=torch.tensor([[0,224,0,224]]).cuda() #0,224,0,224
            locations=[]
            next_locations=[]
            action=0
            with torch.no_grad():
                extract_obs = self.pretrain_network(obs.to(self.device))
            
            for step in range(self.steps_per_episode):

                for i in range(self.action_step):
                    states.append(extract_obs)
                    locations.append(location.clone())
                    with torch.no_grad():
                        policy_k,value=self.get_policy(extract_obs,location/224)
                        if action==1:
                            if i ==0:
                                location[0][0]=location[0][0]+10
                            elif i==1:
                                location[0][1]=location[0][1]-10
                            elif i==2:
                                location[0][2]=location[0][2]+10
                            elif i==3:
                                location[0][3]=location[0][3]-10
                        
                        action,class_pred,action_space,log_probability_actions,train_log=self.get_actions(policy_k,train_log)
                        values.append(value.reshape(1))
                    actions.append(action_space.clone().reshape(1))
                    log_prob_action_k.append(log_probability_actions.reshape(1))

                    next_obs,reward,terminal, info=self.env.step(np.copy(action),class_pred,i,obs)
#                    rew=0.94*rew+reward
                    next_locations.append(location.clone())
                    termination.append(int(terminal[0]))
                    with torch.no_grad():
                        next_obs=F.interpolate(next_obs,size=[obs.shape[2],obs.shape[3]])
                        next_extract_obs = self.pretrain_network(next_obs.to(self.device))
                        action_hoc_x=torch.zeros((1,self.num_class)).cuda()
                        action_hoc_x[:,action_space[0]]=1
                        _, pred_next_state = self.icm(extract_obs, next_extract_obs,action_hoc_x,locations[-1]/224,next_locations[-1]/224)
                        ext_reward=self.fwd_criterion(pred_next_state, torch.cat((self.avgpool(next_extract_obs).squeeze(2).squeeze(2),next_locations[-1]/224),dim=1)) / 2
                    rewards_tol.append(torch.tensor(reward)+ext_reward.cpu())
                    # rewards.append(reward)
                    next_states.append(next_extract_obs)
                    # epoch_log_probability_actions.append(log_probability_action)
                    extract_obs=next_extract_obs
                    obs = next_obs
#                       with torch.no_grad():
#                            logits = self.actor_network(extract_obs)
#                            loss_cls = F.cross_entropy(logits, env._label[sample].cuda())
#                        advantages.append(torch.tensor(reward-0.1).cuda()-loss_cls)
                    if terminal[0]:
                        break_out_flag = True
                        break
                if break_out_flag:
                    break

            #if step>9:
            #    explore_label.append(self.env._label[0])
            values=torch.cat(values)
            sample_reward = []
            advantages = []

            rewards_tol=torch.cat(rewards_tol).cuda()

#            rewards_tol = (rewards_tol) / (rewards_tol.std() + 1e-10)

            with torch.no_grad():
                gae=0
                R=[]
                _, next_value = self.a_c_network(next_extract_obs,location/224)
                next_value=next_value[0]
                for i in reversed(range(len(actions))):
                    gae=gae*self.gamma*args.gae_lambda
                    gae=gae+rewards_tol[i]+self.gamma*next_value.detach()*(1-termination[i])-values[i].detach()
                    next_value=values[i]
                    R.append(gae+values[i])
                R=R[::-1]
                R=torch.cat(R).detach()
                advantages=R-values.detach()
                #advantages=R-torch.mean(R)
            if len(advantages)>1:
                advantages=(advantages-advantages.mean())/(advantages.std()+1e-4)
            else:
                advantages=advantages
            states=torch.cat(states)
            actions=torch.cat(actions)
            log_prob_action_k=torch.cat(log_prob_action_k)

            next_states=torch.cat(next_states).detach()
            locations=torch.cat(locations)
            next_locations=torch.cat(next_locations)

#            advantages=rewards_tol-torch.mean(rewards_tol)

            for i in range(args.update_epochs):
                perm=np.arange(states.shape[0])
                np.random.shuffle(perm)
                
#                states,actions,log_prob_action_k,R,advantages=(
#                    states[perm].clone(),actions[perm].clone(),log_prob_action_k[perm].clone(),
#                    R[perm].clone(),advantages[perm].clone())

                states,actions,log_prob_action_k,advantages,values,next_states,locations,next_locations=(states[perm].clone(),actions[perm].clone(),log_prob_action_k[perm].clone(),
                                                                   advantages[perm].clone(),values[perm].clone(),next_states[perm].clone(),locations[perm].clone(),next_locations[perm].clone())
                for j in range(states.shape[0]//args.update_batch_size):
                    batch_indices=slice(j*args.update_batch_size,min((j+1)*args.update_batch_size,states.shape[0]))
                    policy,critic_values= self.get_policy(states[batch_indices],locations[batch_indices]/224)
                    log_probability_action = policy.log_prob(actions[batch_indices].cuda())
                    log_ratio = log_probability_action - log_prob_action_k[batch_indices]
                    ratio = log_ratio.exp()
                    actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                       torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) *
                                                       advantages[batch_indices]))
                    values_pred = values[batch_indices]+values[batch_indices]*torch.clamp(critic_values.squeeze()/values[batch_indices],1.0-args.epsilon,1.0+args.epsilon)
#                    critic_loss=F.smooth_l1_loss(R[batch_indices],values_pred) ##critic_values.squeeze()
                    critic_loss=F.mse_loss(R[batch_indices],values_pred)
                    entropy_loss = torch.mean(policy.entropy())

                    action_hoc=torch.zeros((states[batch_indices].shape[0],self.num_class)).cuda()
                    action_hoc[list(range(states[batch_indices].shape[0])),actions[batch_indices]]=1
                    pred_action,pred_next_state=self.icm(states[batch_indices],next_states[batch_indices],action_hoc,locations[batch_indices]/224,next_locations[batch_indices]/224)
                    inv_loss=self.inv_criterion(pred_action,actions[batch_indices]) ###action
                    fwd_loss=self.fwd_criterion(pred_next_state,torch.cat((self.avgpool(next_states[batch_indices]).squeeze(),next_locations[batch_indices]/224),dim=1))/2

                    loss=args.l0*(actor_loss+args.c_coef*critic_loss-args.ent_coef*entropy_loss) + (1-args.l1)*inv_loss+args.l1*fwd_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.a_c_network.parameters(), 0.5)
                    self.optimizer.step()

                    lowest = loss < lowest_loss
                    lowest_loss = min(loss, lowest_loss)


            print('epochs:{0} \t' 'batch_reward:{1} \t'.format(episode, torch.mean(rewards_tol)))

            if episode > 5000 or lowest == True:
                self.task ='test'
                acc,pre,label=self.validation_epoch(episode,self.task)
                best=acc>best_prec1
                best_prec1 = max(acc, best_prec1)
                if best:
                    np.save('pred.npy', np.array(pre))
                    np.save('labels.npy', np.array(label))
                    np.save('train_log.npy',np.array(train_log))
                    net_state_dict=self.a_c_network.state_dict()
                    torch.save({
                        'epoch': episode,
                        'net_state_dict': net_state_dict},
                        os.path.join(save_dir, 'best.pt'))
                self.task = 'train'

            episode += 1

        #np.save('explore_label.npy',np.array(explore_label))
        np.save('reward.npy',np.array(epoch_returns))

    def get_policy(self,obs,location):
        logits,value=self.a_c_network.forward(obs,location)

        return Categorical(logits),value
    def get_actions(self, policy,save_log):
        ##action=torch.argmax(policy.logits)
        action_space=policy.sample()
        actions_int=int(action_space.item())
        action=actions_int%3
        class_pred=actions_int//3
        save_log.append([action,class_pred,env._label[0]])
        log_probability_action=policy.log_prob(action_space)
        return int(action),class_pred,action_space,log_probability_action,save_log

    def validation_epoch(self,episode,task):
        self.a_c_network.train(False)
        test_log=[]
        reward,acc,pre,label,test_log=self.test.test(task,test_log)
        np.save('test_log.npy',test_log)
        print('episode:{0} \t' 'batch_reward:{1} \t' 'Acc:{2}'.format(episode,reward[0],acc))
        self.a_c_network.train(True)
        return acc,pre,label



parser = argparse.ArgumentParser(description='PyTorch Tick Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--update-epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-upb', '--update-batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument( '--test-batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.000005, type=float,  #0.000005
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--c_coef', default=1, type=float,
                    metavar='C', help='value loss coeffcient (default: 0.5)')
parser.add_argument('--ent_coef', default=0.01, type=float,
                    metavar='ENT', help='entropy loss coeffcient (default: 0.01)')
parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
parser.add_argument('--l1', default=0.2, type=float,
                    metavar='l1', help='inverse and forward balance (default: 0.2)')
parser.add_argument('--l0', default=0.1, type=float,
                    metavar='l0', help='policy gradient balance (default: 0.1)')
parser.add_argument(
        '--steps_per_episode', help='Maximum steps per episode',
        default=10, type=int)
parser.add_argument(
        '--gae-lambda',
        help="""GAE lambda value""",
        default=0.9, type=float)
parser.add_argument(
        '--task',choices=['train','test'],default='train')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device('cuda:0')
device = torch.device("cuda:0")
traindir = os.path.join(args.data, 'train')
testdir=os.path.join(args.data, 'val')
dataset=Dataset(traindir,testdir,args)
env=TickEnv(dataset,args,image_dim=[224,224])
save_dir=''
train=Training(dataset,env,224,args).train()







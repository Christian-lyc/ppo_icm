
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.distributions.categorical import Categorical
class Test(object):
    def __init__(self,env,image_size,pretrain_model,a_c_network,args,agents,max_steps,action_step=4):

        self.env=env
        self.pretrain_model=pretrain_model
        self.a_c_network=a_c_network
        self.agents=agents
        self.image_size = (image_size, image_size)
        self.action_step=action_step
        self.max_steps=max_steps
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


    def test(self,task,test_log):

        terminal = [False]
        tick_classes=[]
        labels=[]
        class_pred=0
        rewards=0
        policy_p = []

        for steps in range(self.max_steps) :
            break_out_flag = False
            obs = self.env.reset(task)

            extract_obs = self.pretrain_model(obs.to(self.device))
            location=torch.tensor([[0,224,0,224]]).cuda()
            action=0
            for step in range(10):
                for i in range(self.action_step):
                    policy,_=self.a_c_network.forward(extract_obs,location/224)

                    if action==1:
                        if i ==0:
                            location[0][0]=location[0][0]+10
                        elif i==1:
                            location[0][1]=location[0][1]-10
                        elif i==2:
                            location[0][2]=location[0][2]+10
                        elif i==3:
                            location[0][3]=location[0][3]-10
                    #q_values=self.q_network.forward(extract_obs)
                    
#                    class_tick=Categorical(logits=policy)
                    class_tick = policy.max(dim=-1)[1]
                    action=class_tick%3
#                    action_binary = ~(class_tick.to('cpu') == self.env._label[0])
                    class_pred=class_tick//3
                    next_obs, reward, terminal, info = self.env.step(np.copy(action.cpu()),class_pred, i, obs)
                    next_obs = F.interpolate(next_obs, size=[obs.shape[2], obs.shape[3]])
                    next_extract_obs = self.pretrain_model(next_obs.to(self.device))
                    extract_obs = next_extract_obs
                    obs=next_obs
                    rewards+=reward
                    test_log.append([action[0].cpu().numpy(),class_pred[0].cpu().numpy(),self.env._label[0].cpu().numpy()])
                    if terminal[0]:
                        break_out_flag = True
                        break
                if break_out_flag:
                    break
            tick_classes.append(class_pred[0].to('cpu').numpy())
            labels.append(self.env._label[0].numpy())
            policy_p.append(policy.cpu().detach().numpy())
        comparison = [x == y for x, y in zip(tick_classes, labels)]
        count_stats = Counter(comparison)
        acc=count_stats[True]/len(comparison)
        if acc>1:
            np.save('pred.npy', np.array(tick_classes))
            np.save('labels.npy', np.array(labels))
        return rewards/self.max_steps,acc,tick_classes,labels,test_log

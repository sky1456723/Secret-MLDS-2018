from agent_dir.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import collections
import copy
import os
from agent_dir.Model import DQN

np.random.seed(1009)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def prepro(o):
    #print(o.shape)
    o = np.transpose(o, (2,0,1)) # (66, 76, 4), value 0~1
    o = np.expand_dims(o, axis = 0)
    return o 
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            model = torch.load(args.model_name+".ckpt")
            self.current_net = model['current_net']
            self.hyper_param = args.__dict__
            
        elif args.train_dqn:
            if args.load_model:
                model = torch.load(args.model_name+".ckpt")  # dictionary for checkpoint
                self.current_net = model['current_net']
                self.target_net = model['target_net']
                #self.update_target_net()
                self.step_count = 0
                if model['epsilon'] == True:
                    self.epsilon = model['epsilon_value']
                
                self.replay_buffer_len = 10000
                self.replay_buffer = collections.deque([], maxlen = self.replay_buffer_len) 
                self.optimizer = ['Adam', 'RMSprop', 'SGD']
                
                self.training_curve = model['curve']
                self.hyper_param = args.__dict__
                
                if self.hyper_param['optim'] in self.optimizer:
                    if self.hyper_param['optim'] == 'Adam':
                        self.optimizer = torch.optim.Adam(self.current_net.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          betas = (0.9, 0.999))
                    elif self.hyper_param['optim'] == 'RMSprop':
                        self.optimizer = torch.optim.RMSprop(self.current_net.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          alpha = 0.9)
                    elif self.hyper_param['optim'] == 'SGD':
                        self.optimizer = torch.optim.SGD(self.current_net.parameters(),
                                                          lr = self.hyper_param['learning_rate'])
                    #state_dict = torch.load(args.model_name+".optim")
                    #self.optimizer.load_state_dict(state_dict)
                else:
                    print("Unknown Optimizer or missing state_dict!")
                    exit()
                self.env.clip_reward = False
            elif args.new_model:
                if os.path.isfile(args.model_name+".pkl"):
                    # model name conflict
                    confirm = input('Model \'{}\' already exists. Overwrite it? [y/N] ' \
                                    .format(args.model_name.strip('.pkl')))

                    if confirm not in ['y', 'Y', 'yes', 'Yes', 'yeah']:
                        print('Process aborted.')
                        exit()
 
                self.current_net = DQN(84, 84, args.Dueling, args.Noisy)
                self.target_net = DQN(84, 84, args.Dueling, args.Noisy)
                self.update_target_net()
                self.step_count = 0
                if args.epsilon:
                    self.epsilon = 1
                
                self.replay_buffer_len = 10000
                self.replay_buffer = collections.deque([], maxlen = self.replay_buffer_len)
                self.optimizer = ['Adam', 'RMSprop', 'SGD']
                
                self.training_curve = []
                self.hyper_param = args.__dict__
                
                if self.hyper_param['optim'] in self.optimizer:
                    if self.hyper_param['optim'] == 'Adam':
                        self.optimizer = torch.optim.Adam(self.current_net.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          betas = (0.9, 0.999))
                    elif self.hyper_param['optim'] == 'RMSprop':
                        self.optimizer = torch.optim.RMSprop(self.current_net.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          alpha = 0.9)
                    elif self.hyper_param['optim'] == 'SGD':
                        self.optimizer = torch.optim.SGD(self.current_net.parameters(),
                                                          lr = self.hyper_param['learning_rate'])
                else:
                    print("Unknown Optimizer!")
                    exit()
                self.env.clip_reward = False
        print(self.current_net)
        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ### save ckpt as dictionay contain : current_net, target_net, curve, epsilon
        self.current_net = self.current_net.to(device)
        self.target_net = self.target_net.to(device)
        
        batch_size = self.hyper_param['batch_size']
        batch_data = None
        lastest_r = collections.deque([], maxlen = 100)
        for e in range(self.hyper_param['episode']):
            o = self.env.reset()
            o = prepro(o)
            unclipped_episode_r = 0
            while True:
                a = self.make_action(o)
                o_next, r, done, _ = self.env.step(a+1) # map 0,1,2 to 1,2,3
                o_next = prepro(o_next)
                
                unclipped_episode_r += r
                r = np.sign(r)
                #print(r)
                state_reward_tuple = (o, a, r, o_next, done)
                o = o_next
                self.step_count += 1
                if self.hyper_param['epsilon']:
                    self.update_epsilon()
                
                # push to replay buffer
                self.replay_buffer.append(state_reward_tuple)
                
                # get batch data and update current net
                if len(self.replay_buffer) > batch_size and self.step_count%4 == 0:
                    if not self.hyper_param['base']:
                        batch_data = random.sample(self.replay_buffer, batch_size)
                        loss = self.update_param_DDQN(batch_data)  
                        batch_data = None
                        print("Loss: %4f" % (loss), end = '\r')
                # update target net every 1000 step
                if self.step_count % 1000 == 0:
                    print("Update target net")
                    self.update_target_net()
                
                # if game is over, print mean reward and 
                if done:
                    lastest_r.append(unclipped_episode_r)
                    print("Episode : %d Mean : %4f Lastest : %4f" % \
                          (e+1, np.mean(lastest_r), unclipped_episode_r), end = '\n')
                    self.training_curve.append(np.mean(lastest_r))
                    break
                
            unclipped_episode_r = 0
            batch_data = None
            # save model every 500 episode
            if (e+1)%500 == 0:
                self.save_checkpoint(episode = e+1)
            
                
        ##################
        # YOUR CODE HERE #
        ##################
        pass
    
    def update_param_DDQN(self, batch_data):
        """
        batch_data is a list consist of tuple (o_t, a_t, r_t, o_t+1)
        o_t & o_t+1 : numpy array shape of (batch, 4, 66, 76)
        a_t : int
        r_t : int
        """
        
        self.optimizer.zero_grad()
        loss = 0
        if self.hyper_param['Noisy'] and self.hyper_param["Dueling"]:
            self.current_net.value.sample_noise()
            self.target_net.value.sample_noise()
        batch_o_t = []
        batch_o_next = []
        
        batch_r_t = []
        batch_done = []
        '''
        for one_data in batch_data:
            if self.hyper_param['Noisy']:
                current_output = self.current_net(torch.Tensor(one_data[0]).to(device), fixed_noise = True)
                q_t = current_output[0, one_data[1]]
                a_next = self.current_net(torch.Tensor(one_data[3]).to(device), fixed_noise = True)
                a_next = torch.argmax(a_next)
                q_target = self.target_net(torch.Tensor(one_data[3]).to(device), fixed_noise = True)[0, a_next].detach()
            else:
                current_output = self.current_net(torch.Tensor(one_data[0]).to(device))
                q_t = current_output[0, one_data[1]]
                a_next = self.current_net(torch.Tensor(one_data[3]).to(device))
                a_next = torch.argmax(a_next)
                q_target = self.target_net(torch.Tensor(one_data[3]).to(device))[0, a_next].detach()
            loss += F.mse_loss(q_t, 0.99 * (1 - one_data[4]) * q_target+one_data[2])
        '''
        for one_data in batch_data:
            batch_o_t.append(one_data[0])
            batch_o_next.append(one_data[3])
            batch_r_t.append(one_data[2])
            batch_done.append(one_data[4])
        q_t_list = []
        q_target_list = []
        if self.hyper_param['Noisy']:
            current_output = self.current_net(torch.Tensor(batch_o_t).squeeze().to(device), fixed_noise = True)
            a_next = self.current_net(torch.Tensor(batch_o_next).squeeze().to(device), fixed_noise = True)
            a_next = torch.argmax(a_next, dim = 1)
            q_target = self.target_net(torch.Tensor(batch_o_next).squeeze().to(device), fixed_noise = True).detach()
        else:
            current_output = self.current_net(torch.Tensor(batch_o_t).squeeze().to(device))
            a_next = self.current_net(torch.Tensor(batch_o_next).squeeze().to(device))
            a_next = torch.argmax(a_next, dim = 1)
            q_target = self.target_net(torch.Tensor(batch_o_next).squeeze().to(device)).detach()
        for i in range(len(batch_data)):
            q_t_list.append(current_output[i, batch_data[i][1]])
            q_target_list.append( q_target[i, a_next[i]])
        
        q_t_list = torch.stack(q_t_list)
        q_target_list = torch.stack(q_target_list)
        batch_done = torch.Tensor(batch_done).to(device)
        batch_r_t = torch.Tensor(batch_r_t).to(device)
        loss = F.mse_loss(q_t_list, (1-batch_done)*0.99*q_target_list + batch_r_t)
        
        #loss /= self.hyper_param['batch_size']
        loss.backward()
        self.optimizer.step()
        return loss.item()
            
    #need change
    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        # 3 means left, 2 means right, 1 means stay
        # make single action
        
        if not test:
            q_value = self.current_net(torch.Tensor(observation).to(device))
            if self.hyper_param['Noisy']:
                action = torch.argmax(q_value)
                return action.item()
            
            elif not self.hyper_param['Noisy'] and self.hyper_param['epsilon']:
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(3)
                    return action
                else:
                    action = torch.argmax(q_value)
                    return action.item()  
            elif not self.hyper_param['Noisy'] and self.hyper_param['boltzmann']:
                probability = F.softmax(q_value, dim=1)
                random_num = np.random.rand()
                cumulated = 0
                for i in range(probability.shape[1]):
                    cumulated += probability[0, i]
                    if random_num < cumulated:
                        return i
                
        else:
            observation = prepro(observation)
            q_value = self.current_net(torch.Tensor(observation).to(device))
            return torch.argmax(q_value).item()+1
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.current_net.state_dict())
        
    def update_epsilon(self):
        if self.epsilon >= 0.025:
            self.epsilon -= 0.000001
        else:
            pass
        
    def save_checkpoint(self, episode = 0):
        if self.hyper_param['epsilon']:
            e = True
            e_value = self.epsilon
        else:
            e = False
            e_value = None
        check = {'current_net': self.current_net,
                 'target_net': self.target_net,
                 'epsilon': e,
                 'epsilon_value': e_value,
                 'curve': self.training_curve}
        torch.save(check, self.hyper_param['model_name']+"_"+str(episode)+".ckpt")
        
        

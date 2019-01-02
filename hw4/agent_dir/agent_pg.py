import torch
from torch import nn
import torch.nn.functional as F
from agent_dir.agent import Agent
from environment import Environment
import scipy
import numpy as np
import skimage.transform
import os

np.random.seed(1009)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def prepro(o,image_size=[105,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2] #gray scale 
    resized = skimage.transform.resize(y, image_size)[17:-8,:]
    return np.expand_dims(resized.astype(np.float32),axis=2) 

#left to draw curve, unfinished
def draw_curve(agent, env, total_episodes=30, seed = 11037):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    return np.mean(rewards)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)  #with self.env = env in "Agent"
        
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model = torch.load(args.model_name+ ".pkl")
            self.hyper_param = args.__dict__
            self.last_frame = None

        ##################
        # YOUR CODE HERE #
        ##################
        elif args.train_pg:
            self.optimizer = ['Adam', 'RMSprop', 'SGD']
            self.hyper_param = args.__dict__    #make all args to a dict
            self.argument = args    #left to use draw_curve
            self.training_curve = []
            self.last_frame = None  #used to test
            
            if args.new_model:
                if os.path.isfile(args.model_name+".pkl"):
                    # model name conflict
                    confirm = input('Model \'{}\' already exists. Overwrite it? [y/N] ' \
                                    .format(args.model_name.strip('.pkl')))

                    if confirm not in ['y', 'Y', 'yes', 'Yes', 'yeah']:
                        print('Process aborted.')
                        exit()
                
                #initial model
                self.model = nn.Sequential(
                    nn.Linear(80*80*1, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                ).to(device)
                
                #initial optimizer
                if self.hyper_param['optim'] in self.optimizer:
                    if self.hyper_param['optim'] == 'Adam':
                        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          betas = (0.9, 0.999))
                    elif self.hyper_param['optim'] == 'RMSprop':
                        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          alpha = 0.9)
                    elif self.hyper_param['optim'] == 'SGD':
                        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'])
                else:
                    print("Unknown Optimizer!")
                    exit()
                    
            if args.load_model:
                self.model = torch.load(args.model_name+".pkl")
                
                if self.hyper_param['optim'] in self.optimizer:
                    if self.hyper_param['optim'] == 'Adam':
                        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          betas = (0.9, 0.999))
                        state_dict = torch.load(args.model_name+".optim")
                        self.optimizer.load_state_dict(state_dict)
                    elif self.hyper_param['optim'] == 'RMSprop':
                        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'],
                                                          alpha = 0.99)
                        state_dict = torch.load(args.model_name+".optim")
                        self.optimizer.load_state_dict(state_dict)
                    elif self.hyper_param['optim'] == 'SGD':
                        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                          lr = self.hyper_param['learning_rate'])
                else:
                    print("Unknown Optimizer!")
                    exit()       

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.last_frame = None
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        gamma = self.hyper_param['gamma']
        batch = self.hyper_param['batch_size']
        baseline = self.hyper_param['baseline']
        
        #initial some variable to do reward normalization and calculate grad
        total_reward = []
        cumulated_r = []
        p_list = []
        action_list = []
        
        reward_mean = 0
        reward_var = 0
        n_reward = 0
        n_b = 1
        
        loss = 0
        best_result = -21
        lastest_r = []
        
        if not self.hyper_param['PPO']:
            for episode in range(self.hyper_param['episode']):
                
                o = self.env.reset()
                self.last_frame = prepro(o)            
                action = self.env.action_space.sample()
                o, _, _, _ = self.env.step(action)
                episode_r = []
                
                while True:
                    
                    # make_action return the action to do and its probability.
                    # done means the game is over or not, if it's over, then we update
                    # the parameters of model.                  
                    # When reward != 0, we calculate the advantage function using 
                    # discount factor and baseline. And put it to "cumulated_r" according
                    # to each action.
                    
                    o = prepro(o)
                    residual_state = o - self.last_frame   
                    self.last_frame = o
                    action, p = self.make_action(residual_state, test = False)
                    o, reward, done, _ = self.env.step(action)
                    
                    
                    p_list.append(p)                #left for BCELoss
                    action_list.append(action-2)    #left for BCELoss
                    
                    if reward != 0:
                        # When someone gets point, reset discount parameter,
                        # and record the experience
                        # cumulated_r means experience until someone getting point
                        # also calculate mean and var to do reward normalization
                        
                        T = len(cumulated_r)
                        for i in range(T):
                            cumulated_r[i] += (gamma**(T-i))*reward
                        cumulated_r.append(reward)
                        
                        reward_mean += sum(cumulated_r)
                        n_reward += len(cumulated_r)
                        for i in cumulated_r:
                            reward_var += i**2
                        
                        episode_r.append(cumulated_r[-1])
                        total_reward.extend(cumulated_r)
                        cumulated_r = []       
                    else:
                        cumulated_r.append(reward)
                    
            
                    if done and n_b != batch:
                        n_b +=1
                        self.init_game_setting()
                        if len(lastest_r) < 30:
                            lastest_r.append(np.sum(episode_r))
                        elif len(lastest_r) == 30:
                            lastest_r.pop(0)
                            lastest_r.append(np.sum(episode_r))
                            self.training_curve.append(np.mean(lastest_r))
                            #print("Episode : %d Mean : %4f" % (episode, np.mean(lastest_r)), end = ' ')
                        break
                    elif done and n_b == batch:
                        self.init_game_setting()
                        
                        if len(lastest_r) < 30:
                            lastest_r.append(np.sum(episode_r))
                        elif len(lastest_r) == 30:
                            lastest_r.pop(0)
                            lastest_r.append(np.sum(episode_r))
                            self.training_curve.append(np.mean(lastest_r))
                            print("Episode : %d Mean : %4f" % (episode, np.mean(lastest_r)), end = ' ')

                        
                        #update per episode
                        self.optimizer.zero_grad()
                        reward_mean = reward_mean/n_reward
                        reward_stddev = (reward_var/n_reward - (reward_mean)**2)**0.5
                        loss = 0
                        total_reward = torch.Tensor(total_reward).to(device)
                        total_reward = (total_reward - reward_mean) / reward_stddev
                        action_list = torch.Tensor(action_list).to(device)
                        p_list = torch.stack(p_list).view(-1)
                        
                        loss = F.binary_cross_entropy(input = p_list,
                                                      target = action_list,
                                                      weight = total_reward,
                                                      reduction = 'sum')
                        loss /= n_b
                        loss.backward()
                        self.optimizer.step()
                        
                        
                        #reset all record
                        n_b = 1
                        n_reward = 0
                        
                        reward_mean = 0
                        reward_var = 0
                        reward_stddev = 0
                        
                        total_reward = []
                        cumulated_r = []
                        p_list = []
                        action_list = []
                        
                        print("Loss : %4f" % (loss.item()), end = '\r')
                        break
                        
                    
                        
                        
                if episode % 500 == 0 and episode != 0:
                    #Testing per 100 episode
                    print("Testing")
                    test_env = Environment('Pong-v0', self.argument, test=True)
                    result = draw_curve(agent = self, env=test_env)
                    
                    if result >= best_result:
                        best_result = result
                    torch.save(self.model, "./log/"+self.hyper_param['model_name']+".pkl"+str(episode))
                    torch.save(self.optimizer.state_dict(),
                               "./log/"+self.hyper_param['model_name']+".optim")
                    np.save("training_curve.npy", np.array(self.training_curve))
                    print("Save Model, result = ", best_result)
                    
                    #os.system("python3 main.py --test_pg --env_name Pong-v0 -l --model_name " + self.hyper_param['model_name'])
                    
                    print("Finish Testing")
                    
        np.save("training_curve.npy", np.array(self.training_curve))
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
                
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        #action 2 or 4 means up, action 3 or 5 means down
        
        #training
        if not test: 
            observation = torch.Tensor(observation).view(1, -1).to(device)
            output = self.model(observation)
            probability = output[0,0]
            if np.random.rand() < probability.item():
                action = 3
            else:
                action = 2
            return action, output
        #testing
        elif test:
            if type(self.last_frame) == type(None):
                observation = prepro(observation)
                self.last_frame = observation
            else:
                o = prepro(observation)
                observation = o - self.last_frame
                self.last_frame = o
            observation = torch.Tensor(observation).view(1, -1).to(device)
            output = self.model(observation)
            probability = output[0,0]
            if np.random.rand() < probability.item():
                action = 3
            else:
                action = 2
            return action


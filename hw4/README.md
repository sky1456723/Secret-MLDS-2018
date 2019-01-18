# MLDS 2018 HW4  
## Usage
***
```
python3 main.py [-h] [--env_name ENV_NAME] [--train_pg] [--train_dqn]
               [--test_pg] [--test_dqn] [--batch_size BATCH_SIZE]
               [--episode EPISODE] [--learning_rate LEARNING_RATE]
               [--gamma GAMMA] [--baseline BASELINE] [--optim OPTIM] [--PPO]
               [--base] [--model_name MODEL_NAME] [--Dueling] [--Noisy]
               (--load_model | --new_model | --remove_model)
               [--epsilon | --boltzmann]
-h, --help            show this help message and exit
  --env_name ENV_NAME   environment name
  --train_pg            whether train policy gradient
  --train_dqn           whether train DQN
  --test_pg             whether test policy gradient
  --test_dqn            whether test DQN
  
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Specify the batch size for training
  --episode EPISODE, -ep EPISODE
                        Specify how many ep play in an epoch
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Specify the learning rate for training
  --gamma GAMMA         Specify the discount factor of reward
  --baseline BASELINE   Specify the baseline for reward function
  --optim OPTIM         Specify the optimizer, can be Adam, RMSprop, or SGD
  --PPO                 Specify to use PPO or not
  --base                Specify to use base policy gradient or not
  --model_name MODEL_NAME
                        Specify the name of the model
  --Dueling             Specify to use Dueling net or not
  --Noisy               Specify to use NoisyNet net or not
  --load_model, -l      load a pre-existing model
  --new_model, -n       create a new model
  --remove_model, -r    remove a model
  --epsilon             use epsilon greedy
  --boltzmann           use boltzmann exploration  
   
default value and settings in our HW4 for argument:
default              settings
batch_size: 32       (we use batch size = 1 in hw4-1, 32 in hw4-2)
episode: 1000        (we use 5000 in hw4-1, 20000 in hw4-2)
learning_rate: 1e-4  (we use 1e-4 in he4-1, 1.5e-4 in hw4-2)
gamma: 0.99          (we use 0.99 in both hw4-1 and hw4-2)
baseline: 0          (not used)
optim: Adam          (we use both Adam in hw4-1 and hw4-2)
PPO: False           (not used)
base: False          (only used for report to compare the improvement)
model_name: Default_model.pkl  (can be any name you want)
Dueling: False
Noisy: False         (both Dueling and Noisy in hw4-2 improved version)
```  
### To train a new model, you can use like:
```
python3 main.py --train_pg -n --model_name hw4-1 -b 1 -lr 0.0001 -ep 5000
```
to train improved policy gradient (hw4-1)  
```
python3 main.py --train_dqn -n --model_name hw4-2_improved -ep 20000 -b 32 -lr 0.00015 --Dueling --Noisy 
```  
to train improved dqn (hw4-2)  
  
### To test our model, you can clone this directory and use like:  
```
python3 main.py --test_pg -l --model_name ./model/4-1/baseline_model_5000
```
to test our improved policy gradient (hw4-1)  
```
python3 main.py --test_dqn -l --model_name ./model/4-2/test2_dqn_3000_7500 --Dueling --Noisy 
```  
to test our improved dqn (hw4-2)  

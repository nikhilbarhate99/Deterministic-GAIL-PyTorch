import torch
import gym
import numpy as np
from models import GAIL

def train():
    ######### Hyperparameters #########
    env_name = "BipedalWalker-v2"
    random_seed = 0
    n_eval_episodes = 10        # evaluate average reward over n episodes
    lr = 0.0002                 # learing rate
    beta1 = 0.5                 # beta 1 for adam optimizer
    n_epochs = 1000             # number of epochs
    n_iter = 100                # updates per epoch
    batch_size = 100            # num of transitions sampled from expert
    directory = "./preTrained/{}".format(env_name) # save trained models
    filename = "GAIL_{}_{}".format(env_name, random_seed)
    ###################################
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = GAIL(state_dim, action_dim, max_action, lr, beta1)
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # training proceddure
    for epoch in range(1, n_epochs+1):
        # update policy n_iter times
        policy.update(n_iter, batch_size)
        
        # evaluate in environment
        total_reward = 0
        for episode in range(n_eval_episodes):
            state = env.reset()
            for t in range(1000):
                action = policy.select_action(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                
        avg_reward = int(total_reward/n_eval_episodes)
        print("############################")
        print("Epoch: {}\tAvg Reward: {}".format(epoch, avg_reward))
        print("############################")
        
        if avg_reward > 300:
            print("########### Solved! ###########")
            policy.save()
            break
        
if __name__ == '__main__':
    train()

import torch
import gym
import numpy as np
from GAIL import GAIL
import matplotlib.pyplot as plt

def train():
    ######### Hyperparameters #########
    env_name = "BipedalWalker-v2"
    #env_name = "LunarLanderContinuous-v2"
    solved_reward = 300         # stop training if solved_reward > avg_reward
    random_seed = 0
    max_timesteps = 1400        # max time steps in one episode
    n_eval_episodes = 20        # evaluate average reward over n episodes
    lr = 0.0002                 # learing rate
    betas = (0.5, 0.999)        # betas for adam optimizer
    n_epochs = 400              # number of epochs
    n_iter = 100                # updates per epoch
    batch_size = 100            # num of transitions sampled from expert
    directory = "./preTrained/{}".format(env_name) # save trained models
    filename = "GAIL_{}_{}".format(env_name, random_seed)
    ###################################
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)
    
    # graph logging variables:
    epochs = []
    rewards = []
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # training procedure
    for epoch in range(1, n_epochs+1):
        # update policy n_iter times
        policy.update(n_iter, batch_size)
        
        # evaluate in environment
        total_reward = 0
        for episode in range(n_eval_episodes):
            state = env.reset()
            for t in range(max_timesteps):
                action = policy.select_action(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                
        avg_reward = int(total_reward/n_eval_episodes)
        print("Epoch: {}\tAvg Reward: {}".format(epoch, avg_reward))
        
        # add data for graph
        epochs.append(epoch)
        rewards.append(avg_reward)
        
        if avg_reward > solved_reward:
            print("########### Solved! ###########")
            policy.save(directory, filename)
            break
    
    # plot and save graph
    plt.plot(epochs, rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Average Reward')
    plt.title('{}  {}  {} '.format(env_name, lr, betas))
    plt.savefig('./gif/graph_{}.png'.format(env_name))
        
if __name__ == '__main__':
    train()

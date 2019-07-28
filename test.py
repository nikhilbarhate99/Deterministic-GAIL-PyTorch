import gym
from GAIL import GAIL
from PIL import Image

def test():
    env_name = "BipedalWalker-v2"
    #env_name = "LunarLanderContinuous-v2"
    random_seed = 0
    lr = 0.0002
    betas = (0.5, 0.999)
    n_episodes = 3
    max_timesteps = 1000
    render = True
    save_gif = False
    
    directory = "./preTrained/{}".format(env_name)
    filename = "GAIL_{}_{}".format(env_name, random_seed)
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = GAIL(env_name, state_dim, action_dim, max_action, lr, betas)
    
    policy.load(directory, filename)
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
                if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()        
                
if __name__ == '__main__':
    test()
    

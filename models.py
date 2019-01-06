import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ExpertTraj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_label = 1
policy_label = 0

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        
        self.l1 = nn.Linear(state_dim+action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = F.tanh(self.l1(state_action))
        x = F.tanh(self.l2(x))
        x = F.sigmoid(self.l3(x))
        return x
    
    
class GAIL:
    def __init__(self, state_dim, action_dim, max_action, lr, beta1):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(beta1, 0.999))
        
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        self.max_action = max_action
        self.expert = ExpertTraj()
        
        self.loss_fn = nn.BCELoss()
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
        
    def update(self, n_iter, batch_size=100):
        for i in range(n_iter):
            # sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state).to(device)
            exp_action = torch.FloatTensor(exp_action).to(device)
            
            # sample states for actor
            state, _ = self.expert.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = self.actor(state)
            
            #######################
            # update discriminator
            #######################
            self.optim_discriminator.zero_grad()
            
            # with expert transitions
            label = torch.full((batch_size,), exp_label, device=device)
            prob_exp = self.discriminator(exp_state, exp_action)
            loss_exp = self.loss_fn(prob_exp, label)
            loss_exp.backward()
            
            # with policy transitions
            label.fill_(policy_label)
            prob_policy = self.discriminator(state, action.detach())
            loss_policy = self.loss_fn(prob_policy, label)
            loss_policy.backward()
            
            self.optim_discriminator.step()
            
            ################
            # update policy
            ################
            self.optim_actor.zero_grad()
            
            label.fill_(exp_label)
            prob_actor = self.discriminator(state, action)
            loss_actor = self.loss_fn(prob_actor, label)
            loss_actor.backward()
            
            self.optim_actor.step()
            
        
            
            
            
            
        
        
    
        
    
    def save(self, directory='./preTrained', name='GAIL'):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory,name))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory,name))
        
    def load(self, directory='./preTrained', name='GAIL'):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory,name)))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory,name)))
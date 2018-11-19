from models import Actor, Critic 
from torch.optim import Adam
import torch
import numpy as np

from OUnoise import OUNoise

device = 'cpu'

LR_ACTOR = 9e-5         # learning rate of the actor 
LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay


class DDPGAgent():
    """
    Agent that interacts with and learns from the environment.
    
    """
    
    def __init__(self, state_size, action_size, agent_num, random_seed):
        """
        Initialize an Agent object.
        :param state_size (int): dimension of each state
        :param action_size (int): dimension of each action
        :param random_seed (int): random seed
        """

        # Actor Networks 
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Networks 
        self.critic_local = Critic(state_size, action_size, agent_num, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, agent_num, random_seed).to(device)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed, scale=0.1)
        
    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor_local(obs) + noise*self.noise.sample()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor_target(obs) + noise*self.noise.sample()
        return action

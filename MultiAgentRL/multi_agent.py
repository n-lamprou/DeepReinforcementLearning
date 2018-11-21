from utils import soft_update, transpose_to_tensor, transpose_list
from collections import deque, namedtuple
import random
import torch
import numpy as np

device = 'cpu'

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 2       # how often to update the network
UPDATE_BATCHES = 1     # number of minibatch updates 


class MADDPG:
    def __init__(self, agents):
        super(MADDPG, self).__init__()

        self.maddpg_agent = agents
        self.num_agents = len(agents)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=0)
        self.t_step = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [agent.actor_local for agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [agent.actor_target for agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        a_s = zip(self.maddpg_agent, obs_all_agents.view(self.num_agents, BATCH_SIZE, -1))
        target_actions = torch.stack([agent.target_act(obs, noise) for agent, obs in a_s]).view(BATCH_SIZE, self.num_agents, -1)
        return target_actions
    
    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > 2*BATCH_SIZE:
                for i in range(UPDATE_BATCHES):
                    for a_i in range(self.num_agents):
                        experiences = self.memory.sample()
                        # Update Critic and Actor of each agent
                        self.learn(experiences, a_i)
                        # Update target networks
                        self.update_targets()

    def learn(self, samples, agent_number):
        """update the critics and actors of all the agents """

        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
        
        agent = self.maddpg_agent[agent_number]
        
        # ---------------------------- update Critic ---------------------------- #
        agent.critic_optimizer.zero_grad()

        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(list(target_actions.t()), dim=1)
        
        target_critic_input = torch.cat((next_obs_full.view(BATCH_SIZE, -1), target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.critic_target(target_critic_input)
        
        y = reward.t()[agent_number].view(-1, 1) + GAMMA * q_next * (1 - done.t()[agent_number].view(-1, 1))
        action = torch.cat(list(action.t()), dim=1)
        critic_input = torch.cat((obs_full.view(BATCH_SIZE, -1), action), dim=1).to(device)
        q = agent.critic_local(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 0.5)
        agent.critic_optimizer.step()

        # ---------------------------- update Actor ---------------------------- #
        agent.actor_optimizer.zero_grad()

        q_input = [ self.maddpg_agent[i].actor_local(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor_local(ob).detach()
                   for i, ob in enumerate(obs.t())]
        
        q_input = torch.stack(q_input).t()
        q_input = torch.cat(list(q_input.t()), dim=1)
        q_input2 = torch.cat((obs_full.view(BATCH_SIZE, -1), q_input), dim=1)
        
        actor_loss = -agent.critic_local(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        
    def update_targets(self):
        """soft update targets"""
        for agent in self.maddpg_agent:
            soft_update(agent.actor_target, agent.actor_local, TAU)
            soft_update(agent.critic_target, agent.critic_local, TAU)
    
    
class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.
        :param buffer_size (int): maximum size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.stack(transpose_to_tensor(transpose_list([e.state for e in experiences if e is not None])))
        states_full = states.view(self.batch_size, 1, -1)               
        actions = torch.stack(transpose_to_tensor(transpose_list([e.action for e in experiences if e is not None])))
        rewards = torch.stack(transpose_to_tensor(transpose_list([e.reward for e in experiences if e is not None])))
        next_states = torch.stack(transpose_to_tensor(transpose_list([e.next_state for e in experiences if e is not None])))
        next_states_full = next_states.view(self.batch_size, 1, -1)                     
        dones = torch.stack(transpose_to_tensor(transpose_list([e.done for e in experiences if e is not None])))

        return (states, states_full, actions, rewards, next_states, next_states_full, dones)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)

       

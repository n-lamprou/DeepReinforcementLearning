from collections import deque 
import numpy as np
import torch 
from utils import transpose_to_tensor, transpose_list

# import environment 
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name='Tennis.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# environment details
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

# import models and agents
from models import Actor, Critic 
from agents import DDPGAgent
from multi_agent import MADDPG


def run(agent):
    """
    Run simulation    
    :param agent: Agent to use to determine actions 
    """
    
    # Begin simulation
    env_info = env.reset(train_mode=False)[brain_name]      # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]      # reset the environment
    states = env_info.vector_observations                   # get the current states
    scores = np.zeros(num_agents)                           # initialize the score
    while True:
        states_ = transpose_to_tensor(transpose_list(states))
        actions = maddpg.act(states_, noise=0)               # select actions (without noise)
        actions = torch.stack(actions).detach().numpy()
        env_info = env.step(actions)[brain_name]            # send the actions to the environment
        next_states = env_info.vector_observations          # get the next states
        rewards = env_info.rewards                          # get the reward
        dones = env_info.local_done                         # see if episode has finished
        scores += rewards                                   # update the scores
        states = next_states                                # roll over the states to next time step
        if np.any(dones):                                   # exit loop if episode finished
            break
    
    print("Score: {}".format(np.max(scores)))

    
if __name__ == "__main__":
    
    # instantiate agent and load weights
    Player1 = DDPGAgent(state_size, action_size, num_agents, random_seed=0)
    Player2 = DDPGAgent(state_size, action_size, num_agents, random_seed=0)
    
    maddpg = MADDPG(agents=[Player1, Player2])
    
    for i, a in enumerate(maddpg.maddpg_agent):
        checkpoint = torch.load('MADDPG_actor_{}.pth'.format(i+1))
        a.actor_local.load_state_dict(checkpoint)
        
    # Run simulation with specified agent
    print('Running simulation with MADDPG agent')
    run(maddpg)
    env.close()


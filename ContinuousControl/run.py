from collections import deque 
import numpy as np
import torch 

# import environment 
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name='Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# import models and agents
from models import Actor, Critic 
from agents import DDPGAgent


def run(agent):
    """
    Run simulation    
    :param agent: Agent to use to determine actions 
    """
    
    # Begin simulation
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, add_noise=False)    # select an action (without noise)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
    
    print("Score: {}".format(score))

    
if __name__ == "__main__":
    
    # instantiate agent and load weights
    agent = DDPGAgent(state_size=33, action_size=4, model=(Actor, Critic), random_seed=0)
    agent.actor_local.load_state_dict(torch.load('DDPG_actor.pth'))
        
    # Run simulation with specified agent
    print('Running simulation with DDPG agent')
    run(agent)
    env.close()


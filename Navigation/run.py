import argparse
from collections import deque 
import numpy as np
import sys
import os
import torch 

# import environment 
sys.path.append('../python/')
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# import models and agents
from models import QNetwork, DuelingQNetwork
from agents import DQNAgent, DDQNAgent


def run(agent, name):
    """
    Run simulation
    
    Params
    ======
        agent (object): Agent to train
        name (string): name of agent for loading model parameters
    """
    # Load model parameters into agent
    checkpoint = torch.load('{}_checkpoint.pth'.format(name))
    agent.qnetwork_local.load_state_dict(checkpoint)
    
    # Begin simulation
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = np.int32(agent.act(state, 0))         # select an action (setting epsilon to zero)
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
    
   # Set up arguement parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-ddqn", "--double_dqn", help="Boolean - Whether to use double deep Q network")
    parser.add_argument("-duel", "--duelling", help="Boolean - Whether to use duelling architecture")
    args = parser.parse_args()
    
    # Set default parameters
    if args.double_dqn == 'True':
        args.double_dqn = True
    else: 
        args.double_dqn = False
    if args.duelling == 'True':
        args.duelling = True
    else: 
        args.duelling = False

    print("Double DQN {}, Duelling Architecture {}".format(args.double_dqn, args.duelling))
    
    # instantiate appropriate agent
    if (args.double_dqn is True) & (args.duelling is True):
        agent = DDQNAgent(state_size=37, action_size=4, model=DuelingQNetwork, seed=0)
        agent_name = 'duel_ddqn'
    
    elif (args.double_dqn is True) & (args.duelling is False):
        agent = DDQNAgent(state_size=37, action_size=4, model=QNetwork, seed=0)
        agent_name = 'ddqn'
    
    elif (args.double_dqn is False) & (args.duelling is True):
        agent = DQNAgent(state_size=37, action_size=4, model=DuelingQNetwork, seed=0)
        agent_name = 'duel_dqn'
        
    else:
        agent = DQNAgent(state_size=37, action_size=4, model=QNetwork, seed=0)
        agent_name = 'dqn'
        
    # Run simulation with specified agent
    print('Running simulation with {} agent'.format(agent_name))
    run(agent, agent_name)
    env.close()


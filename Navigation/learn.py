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


def learn(agent, name, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning
    
    Params
    ======
        agent (object): Agent to train
        name (string): name of agent for saving model parameters
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = np.int32(agent.act(state, eps))
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # get whether episode complete
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
                
        # save most recent score
        scores_window.append(score)       
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=15:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '{}_checkpoint.pth'.format(name))
            break

            
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
        
    # Learn agent and save model
    print('Training {} agent'.format(agent_name))
    learn(agent, agent_name)
    env.close()

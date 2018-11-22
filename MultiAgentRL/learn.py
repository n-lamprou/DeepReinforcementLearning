from collections import deque 
import numpy as np
import torch 
from utils import transpose_to_tensor, transpose_list

## import environment 
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

            
def learn(agent, n_episodes=5000):
    """
    Train an intelligent agent to interact with environment
    
    :param agent (object): Agent to train
    :param n_episodes (int): maximum number of training episodes
    """ 
    
    NOISE = 2
    NOISE_REDUCTION = 0.999

    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, 10000):
        env_info = env.reset(train_mode=True)[brain_name]               # reset the environment
        states = env_info.vector_observations                           # get state for each agent
        scores = np.zeros(num_agents)
        while True:
            states_ = transpose_to_tensor(transpose_list(states))
            actions = agent.act(states_, noise=NOISE)                   # get actions for each agent
            NOISE *= NOISE_REDUCTION
            actions = torch.stack(actions).detach().numpy()
            env_info = env.step(actions)[brain_name]                    # send all actions to tne environment
            next_states = env_info.vector_observations                  # get next state (for each agent)
            rewards = env_info.rewards                                  # get reward (for each agent)
            scores += env_info.rewards
            dones = env_info.local_done                                 # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)    # perform learning step
            states = next_states
            if np.any(dones):                                           # exit loop if episode finished
                break

        # save most recent score
        scores_window.append(round(np.max(scores),2)) 
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
   
        if np.mean(scores_window) >= 1:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            for i, a in enumerate(agent.maddpg_agent):
                torch.save(a.actor_local.state_dict(), 'MADDPG_actor_{}.pth'.format(i+1))
            break

            
if __name__ == "__main__":
    
    # instantiate agents and multiagent framework
    Player1 = DDPGAgent(state_size, action_size, num_agents, random_seed=0)
    Player2 = DDPGAgent(state_size, action_size, num_agents, random_seed=0)
    
    maddpg = MADDPG(agents=[Player1, Player2])
        
    # Learn agent and save model
    learn(maddpg)
    env.close()

import argparse
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

            
def learn(agent, n_episodes=500, noise=True):
    """
    Train an intelligent agent to interact with environment
    
    :param agent (object): Agent to train
    :param n_episodes (int): maximum number of training episodes
    :param noise (bool): whether OU noise should be used in training
    """
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, add_noise=noise)
            env_info = env.step(action)[brain_name]              # send the action to the environment
            next_state = env_info.vector_observations[0]         # get the next state
            reward = env_info.rewards[0]                         # get the reward
            done = env_info.local_done[0]                        # get whether episode complete
            agent.step(state, action, reward, next_state, done)  # perform learning step
            state = next_state
            score += reward
            if done:
                break 
                
        # save most recent score
        scores_window.append(round(score,2)) 
        
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
   
        if np.mean(scores_window)>=30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'DDPG_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'DDPG_critic.pth')
            break

            
if __name__ == "__main__":
    
    # Set up arguement parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-OUnoise", "--OU_noise", help="Boolean - whether to add OU noise to action vector during training")
    args = parser.parse_args()
    
    # Set default parameters
    if args.OU_noise == 'False':
        args.OU_noise = False
        print('Training DDPG agent without OU noise')
    else: 
        args.OU_noise = True
        print('Training DDPG agent with OU noise')
    
    # Instantiate Agent
    agent = DDPGAgent(state_size=33, action_size=4, model=(Actor, Critic), random_seed=0)
        
    # Learn agent and save model
    learn(agent, noise = args.OU_noise)
    env.close()

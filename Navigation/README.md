# Deep Reinforcement Learning - Navigation


## Project Details

The aim of this project is to train an agent to navigate and collect yellow bananas in a large, square world. The task is episodic, and in order to solve the environment, the agent must get an average score of +15 over 100 consecutive episodes.


#### State 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. 


#### Actions

Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.


#### Rewards

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas in a given time frame.


#### Expected Behaviour

<img src="images/NavigationProject_Medium.gif" width="800" height="500" />



## Getting Started


#### Step 1: Clone the Repository

You will need a python 3.6 environment set up. To be able to train and run the agents, you will need the install the dependencies. Do do so, clone the repostory and install the required packages. With the following commands you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required.

```bash
git clone https://github.com/n-lamprou/DeepReinforcementLearning.git
cd DeepReinforcementLearning/python
pip install .
```


#### Step 2: Download the Unity Environment

The repository already has the Windows the Banana environment built and placed in the directory `Banana_Windows_x86_64`. To run on linux or mac you can download the built environment following the links that matches your operating system:

* Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

Replace the Place the `Banana_Windows_x86_64` directory in the `Navigation/` folder repository with teh corresponding unzipped directory of your choice.


## Instructions

Once your environment is set up, navigate to the `Navigation` folder and follow the instructions beneath to train an agent or to run a simulation. 

#### Training an Agent

To train an agent run the `learn.py` script. By default this trains a DQN agent (see report for more details). You can instead choose whether you want the agent to use a Double DQN algorithm or a Dueling architecture or both. For example to train an agent with both improvements run the following command:

```bash
python learn.py -ddqn True -duel True
```

For more information on the options available run

```bash
python learn.py -h
```

#### Running a simulation

To run a simulation with you agent of choice run the following command:

```bash
python run.py -ddqn True -duel True
```

This will open a window showing your agent navigating the banana environment. 4 pre-trained agents are included in the repo.

Enjoy!
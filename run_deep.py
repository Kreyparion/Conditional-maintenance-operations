# ENV
from project.env.environnement import Environnement

# AGENT
from project.agents.agent import Agent
from project.implemented_agents import agents_map
from project.agents.SARSA import EpsilonGreedyAgent
from project.agents.SARSA_expected import EpsilonGreedyAgent_expected
from project.agents.Qlearning import EpsilonGreedyAgent_Qlearning
from project.agents.random import RandomAgent
from project.agents.deep import DeepLearningAgent, mlp



# PYTHON
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torchtyping import TensorType
from project.agents.agent import Agent
from project.env.environnement import Environnement
from project.env.states import State
from project.env.actions import Action
from random import random, choice


batch_size = 5
N_epoch = 1000

                
# for training policy
def train_one_epoch(agent : Agent, env : Environnement,state):
    
    batch = 0
    
    done = False
    
    agent.batch_reset()
    
    while not done:
        # Agent takes action
        action = agent.act(state)   # on prend l'action avec l'observation qu'on vient de voir

        # Action has effect on environment
        
        next_state, reward, done = env.step(action)

        # Agent observe the transition and possibly learns
        agent.observe(state, action, reward, next_state, done)
        
        batch += 1
        
        if batch == batch_size:      # agent learns
            batch = 0
            done = True
            
            
        # Render environment for user to see
        env.render()

        # Update state
        state = next_state
    
    print("reward:", reward)
    print("action", action)

    agent.learn()
    return state
            
     
# Create the environnement
env = Environnement.init("3d")

# Create the agent
agent = DeepLearningAgent(env,hidden_sizes=[32],lr=1e-2,batch_size=batch_size,render=False)
    

def train(N_epoch):
    agent.reset()
    state = env.reset()   # first obs/state comes from starting distribution 
    for i in range(N_epoch):
        print("Epoch:" +str(i))
        state = train_one_epoch(agent,env,state)


train(N_epoch)

# HEY

"""


if __name__ == "__main__":
    
    # Get args

    parser = ArgumentParser(description="Run a reinforcement learning agent")
    parser.add_argument("--agent", type=str, required=True, help="Agent to run")
    args = parser.parse_args()
    agent_name = args.agent

    # Create the environnement
    env = Environnement.init("3d")

    # Create the agent
    agent = agents_map[agent_name](env)
    # Run the agent
    print(agent)
    train(agent, env)

"""
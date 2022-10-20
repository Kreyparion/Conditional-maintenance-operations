from tp.agents.agent import Agent,ValueBasedAgent
from tp.utils import State,Action
from random import choice,random, randint
import gym
import numpy as np


class TD_EGreedyAgent(ValueBasedAgent):
    
    def __init__(self, env : gym.Env, **kwargs):
        self.env = env
        self.stateCrossActions = [[(state, action) for action in env.gridworld.getPossibleActions(state)] for state in env.mdp.getStates()][1:]
        self.allstates = env.mdp.getStates()[1:]
        # Hyperparameters
        self.GAMMA = 0.99
        self.ALPHA = 0.1
        self.EPSILON = 0.1
        
        
        self.previous_action = None
        self.previous_state = None
        
        self.current_action = None
        self.current_state = None
        self.current_reward = 0

        self.done = False
        
        # Init QValue
        self.qvalue = dict()
        for state in env.mdp.getStates()[1:]:
            self.qvalue[state] = dict()
            for action in env.gridworld.getPossibleActions(state):
                self.qvalue[state][action] = 0


    def act(self, state: State, training = None):
        # use policy to act at a certain state
        return
    
    def maxi_action(self, state: State):
        maxi = -100000
        action_maxi = None
        for a,x in self.qvalue[state].items():
            if x >= maxi:
                action_maxi = a
                maxi = x
        return maxi, action_maxi
    
    def policy(self,state:State):
        # define Epsilon greedy policy
        return
        
    
    def observe(self, state: State, action: Action, reward, next_state: State, done):
        self.current_action = action
        self.current_state = state
        self.current_reward = reward
        self.done = done

    def learn(self):
        if self.previous_state != None:
            self.qvalue[self.previous_state][self.previous_action] = self.qvalue[self.previous_state][self.previous_action] + self.ALPHA*(self.previous_reward+self.GAMMA*self.qvalue[state][action] - self.qvalue[self.previous_state][self.previous_action])

    
    def getQValue(self, state: State, action: Action) -> float:
        return self.qvalue[state][action]


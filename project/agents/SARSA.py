from project.agents.agent import Agent
from project.env.environnement import Environnement
from project.env.states import State
from project.env.actions import Action
from project.tools.logger import logger, init_logger
from random import random,choice


class EpsilonGreedyAgent(Agent):
    
    def __init__(self, env : Environnement, **kwargs):
        self.env = env
        init_logger(logger)
        self.stateCrossActions = [[(state, action) for action in env.getPossibleActions(state)] for state in env.getEveryState()]
        # self.allstates = env.mdp.getStates()[1:] # Give the list of all states
        # Hyperparameters
        self.GAMMA = 0.99
        self.ALPHA = 0.1
        self.EPSILON = 0.1
        
        
        self.previous_action = None
        self.previous_state = None
        self.previous_reward = 0
        
        self.current_action = None
        self.current_state = None
        self.current_reward = 0

        self.done = False
        
        # Init QValue
        self.qvalue = dict()
        for state in env.getEveryState():
            self.qvalue[state] = dict()
            for action in env.getPossibleActions(state):
                self.qvalue[state][action] = 0


    def act(self, state: State, training = None):
        # use policy to act at a certain state         
        return self.policy(state)
    
    def maxi_action(self, state: State): # Give the maximum action value in a state and the corresponding action
        maxi = -100000
        action_maxi = None
        for a,x in self.qvalue[state].items():
            if x >= maxi:
                action_maxi = a
                maxi = x
        return maxi, action_maxi
    
    def policy(self,state:State):
        # define Epsilon greedy policy
        proba = random()
        if proba < self.EPSILON:
            action = choice(list(self.qvalue[state].keys()))   # choice of a random action
        else:
            maxi,action = self.maxi_action(state)
        return action
        

    def observe(self, state: State, action: Action, reward, next_state: State, done):
        self.current_action = action
        self.current_state = state
        self.current_reward = reward
        self.done = done
        logger.info(f"Step : {self.env.step_number-1} Agent observes: state={state}, action={action}, reward={reward}, next_state={next_state}, done={done}")

    def learn(self):
        #Learn
        if self.env.step_number % 500 == 499:
            self.EPSILON *= 0.999
        if self.previous_state == None:
            pass
        else:
            current_Qvalue = self.getQValue(self.current_state,self.current_action)
            previous_Qvalue = self.getQValue(self.previous_state,self.previous_action)
            previous_Qvalue = previous_Qvalue + self.ALPHA*(self.previous_reward + self.GAMMA*current_Qvalue - previous_Qvalue)
            self.qvalue[self.previous_state][self.previous_action] = previous_Qvalue
            
            if self.done == True:
                # Learn
                current_Qvalue = current_Qvalue + self.ALPHA*(self.current_reward - current_Qvalue)
                self.qvalue[self.current_state][self.current_action] = current_Qvalue
                
                   
        self.previous_action = self.current_action
        self.previous_state = self.current_state
        self.previous_reward = self.current_reward
        
        
    def getQValue(self, state: State, action: Action) -> float:
        return self.qvalue[state][action]
    
    
    def reset(self):
        self.previous_action = None
        self.previous_state = None
        self.previous_reward = 0
        
        self.current_action = None
        self.current_state = None
        self.current_reward = 0
from project.agents.agent import Agent
from project.env.environnement import Environnement
from project.env.states import State
from project.env.actions import Action
from random import random,choice
import gym


class EpsilonGreedyAgent_memory(Agent):
    
    def __init__(self, env : gym.Env, **kwargs):
        self.env = env
        self.stateCrossActions = [[(state, action) for action in env.getPossibleActions(state)] for state in env.getEveryState()][1:]
        # self.allstates = env.mdp.getStates()[1:] # Give the list of all states
        # Hyperparameters
        self.GAMMA = 0.99
        self.ALPHA = 0.1
        self.EPSILON = 0.1
        
        
        self.current_action = None
        self.current_state = None
        self.current_reward = 0

        self.done = False
        
         # Init QValue
        self.qvalue = dict()
        for state in env.getEveryState()[1:]:
            self.qvalue[state] = dict()
            for action in env.getPossibleActions(state):
                self.qvalue[state][action] = 0
                
        # Memory of the 9 last qvalue
        self.a_memo = []
        self.s_memo = []
        self.r_memo = []
        self.oldest = 0


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

        
    def learn(self):
        #Learn
        
        if len(self.r_memo) == 0:
            pass
        
        else:
            current_Qvalue = self.getQValue(self.current_state,self.current_action)    # à t+10
            oldest_Qvalue = self.getQValue(self.s_memo[self.oldest],self.a_memo[self.oldest])    # à t
            
            r_sum = 0
            for i in range(len(self.r_memo)):
                r_sum += self.r_memo[i] 
                            
            oldest_Qvalue = oldest_Qvalue + self.ALPHA*(r_sum + self.GAMMA**10*current_Qvalue - oldest_Qvalue)
            self.qvalue[self.s_memo[self.oldest]][self.a_memo[self.oldest]] = oldest_Qvalue
            
            if self.done == True:
                # Learn
                self.r_memo.pop(self.oldest)
                
                while len(self.r_memo) != 0:
                    self.oldest = (self.oldest + 1) % 10
                    r_sum = 0
                    for i in range(len(self.r_memo)):
                        r_sum += self.r_memo[i]     
                    oldest_Qvalue = oldest_Qvalue + self.ALPHA*(r_sum + self.GAMMA**10*current_Qvalue - oldest_Qvalue)
                    self.qvalue[self.s_memo[self.oldest]][self.a_memo[self.oldest]] = oldest_Qvalue
                    self.r_memo.pop(self.oldest)                    
    
            
            if len(self.r_memo) < 9:
                self.r_memo = self.r_memo * self.GAMMA
                self.r_memo.append(self.current_reward)
            else:
                self.r_memo = self.r_memo * self.GAMMA
                self.a_memo[self.oldest] = self.current_action
                self.s_memo[self.oldest] = self.current_state
                self.r_memo[self.oldest] = self.current_reward
                self.oldest = (self.oldest + 1) % 10

        
    
    def getQValue(self, state: State, action: Action) -> float:
        return self.qvalue[state][action]
    
    def reset(self):
        self.current_action = None
        self.current_state = None
        self.current_reward = 0
        self.a_memo = []
        self.r_memo = []
        self.s_memo = []
        self.oldest = 0

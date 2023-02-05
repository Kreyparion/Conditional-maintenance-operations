from project.agents.agent import Agent
from project.env.environnement import Environnement
from project.env.states import State
from project.env.actions import Action
from project.tools.logger import logger, init_logger
from random import random,choice,randint
import json

class EpsilonGreedyAgent_Qlearning(Agent):
    
    def __init__(self, env : Environnement, **kwargs):
        self.env = env
        self.stateCrossActions = [[(state, action) for action in env.getPossibleActions(state)] for state in env.getEveryState()]
        init_logger(logger)
        logger.info(f"Size of State-Action space {(len(self.stateCrossActions),len(self.stateCrossActions[0]))}")
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
        
        self.no_act_count = 0
        
         # Init QValue
        self.qvalue = dict()
        for state in self.env.getEveryState():
            self.qvalue[state] = dict()
            for action in self.env.getPossibleActions(state):
                self.qvalue[state][action] = 0
        
        #self.load_agent()


    def act(self, state: State, training = None):
        # use policy to act at a certain state
        if (self.previous_state == state and self.previous_state != self.env.out_of_order_state()) or self.env.action_in_queue:
            return Action.ActionDoNothing()
        else:
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
        if self.no_act_count > 0:
            self.no_act_count -= 1
            return Action.ActionDoNothing()
        proba = random()
        if proba < self.EPSILON:   
            action = choice(list(self.qvalue[state].keys()))   # choice of a random action
            if action == Action.ActionDoNothing():
                self.no_act_count = 0
        else:
            maxi,action = self.maxi_action(state)
        return action
    
    
    def observe(self, state: State, action: Action, reward, next_state: State, done):
        self.current_action = action
        self.current_state = state
        self.current_reward = reward
        self.done = done
        if done:
            self.EPSILON = 0.08



        if action != Action.ActionDoNothing():
            logger.info(f"Step : {self.env.step_number-1} Agent observes: state={state}, action={action}, reward={reward}, done={done}")
            logger.info(f"State value initial_step {self.maxi_action(self.env.initial_state())}")
            logger.info(f"State value for state {self.qvalue[state]}")
    
    
    def save_agent(self):
        def to_json(json_data):
            new_json_data = dict()
            for state, actions in json_data.items():
                state_tuple = state.to_tuple()
                state_str = "state : " + str(state_tuple)
                new_json_data[state_str] = dict()
                for action, value in actions.items():
                    action_tuple = action.to_tuple()
                    action_str = "action : " + str(action_tuple)
                    new_json_data[state_str][action_str] = value
            return new_json_data
        with open('qvalue.json', 'w+') as fp:
            json.dump(to_json(self.qvalue), fp, indent=4)
    
    def load_agent(self):
        def from_json(json_data):
            new_json_data = dict()
            for state, actions in json_data.items():
                state_tuple = state[9:-1].split(", ")
                state_list = [int(x) for x in state_tuple]
                state = self.env.get_state_with_wear(state_list)
                new_json_data[state] = dict()
                for action, value in actions.items():
                    action_tuple = action[10:-1].split(", ")
                    action_tuple = [int(x) for x in action_tuple]
                    action = Action.fromListInt(action_tuple)
                    new_json_data[state][action] = value
            return new_json_data
        with open('qvalue.json', 'r') as fp:
            self.qvalue = from_json(json.load(fp))
    
    
    def learn(self):
        #Learn
        if self.env.step_number % 500 == 499:
            self.EPSILON *= 0.99
            logger.info(f"Step : {self.env.step_number-1} Agent learns: EPSILON={self.EPSILON}")
        #Save the agent
        if self.env.step_number % 100000 == 99999:
            self.save_agent()
        if self.previous_state == None:
            pass
        else:
            Q_greedy,action_greedy = self.maxi_action(self.current_state)
            current_Qvalue = self.getQValue(self.current_state,self.current_action)
            previous_Qvalue = self.getQValue(self.previous_state,self.previous_action)
            previous_Qvalue = previous_Qvalue + self.ALPHA*(self.previous_reward + self.GAMMA*Q_greedy - previous_Qvalue)
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

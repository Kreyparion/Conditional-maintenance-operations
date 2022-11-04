from project.agents.agent import Agent
from project.env.environnement import Environnement
from project.env.states import State
from project.env.actions import Action
from random import choice



class RandomAgent(Agent):
    def __init__(self, env: Environnement, **kwargs):
        super().__init__(env, **kwargs)
        self.env = env

    def act(self, state:State):
        return choice(self.env.getPossibleActions(state))

    def observe(self, state:State, action:Action, reward:float, next_state:State, done:bool):
        pass

    def learn(self):
        pass
    
    def random(self):
        pass
    
    def reset(self):
        pass
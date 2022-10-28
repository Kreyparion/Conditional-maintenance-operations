from abc import ABC, abstractmethod

from project.env.environnement import Environnement
from project.env.actions import Action
from project.env.states import State

class Agent(ABC):
    """
    Base class for all of our model-based agents. 
    An agent is an object that can interact with an environment and learn from it.
    """
    def __init__(self, env : Environnement, **kwargs):
        self.env = env

    @abstractmethod
    def act(self, state: State) -> Action:
        """Return the action to take in the given state"""
        pass

    @abstractmethod
    def observe(self, state: State, action: Action, reward: float, next_state: State, done: bool):
        """Observe the transition and stock in memory"""
        pass

    @abstractmethod
    def learn(self):
        """Learn using the memory"""
        pass   
    
    @abstractmethod
    def reset(self):
        """Learn using the memory"""
        pass    
 


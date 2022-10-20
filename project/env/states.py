from project.env.items import Item
from typing import List, Dict

class State:
    """Surely a list of items"""
    def __init__(self,continuous:bool) -> None:
        self.continuous = continuous

    @classmethod
    def get_states(self)-> List["State"]:
        """Method that returns all possible states in case the environment is discrete"""
        pass




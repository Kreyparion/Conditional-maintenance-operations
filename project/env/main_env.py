from enum import Enum
from typing import List,Dict,Union

from gym import Env
from gym.spaces import Discrete, Space

from project.env.items import Item




class CoreAction:
    """unit actions on items, listed in _list_actions ("preventive","corrective")

    Raises:
        NameError: the action is not implemented
    """
    _list_actions = ["preventive","corrective"]
    def __init__(self,action_type:str) -> None:
        if action_type == self._list_actions[0]:
            self.coreaction = self._list_actions[0]
        elif action_type == self._list_actions[1]:
            self.coreaction = self._list_actions[1]
        else:
            raise NameError("Action not existing")

    @classmethod
    def fromNumber(self,i:int) -> 'CoreAction':
        """create an object from an integer (the order is defined by _list_actions) : 0 for "preventive", 1 for"corrective"

        Args:
            i (int): the integer referencing the action

        Raises:
            IndexError: the index isn't implemented in the known action list

        Returns:
            CoreAction: The associated object 
        """
        if i == 0:
            return CoreAction(self._list_actions[0])
        elif i == 1:
            return CoreAction(self._list_actions[1])
        else:
            raise IndexError("Index not corresponding to any Action")
    
    @classmethod
    @property
    def listCoreActions(self) -> List['Action']:
        """Generate the list of all Coreaction

        Returns:
            List[Action]: The list of all possible unit actions
        """
        return [CoreAction(x) for x in self._list_actions]
    
    def __eq__(self, other: 'CoreAction'):
        return self.coreaction == other.coreaction
    
    def __repr__(self):
        return self.coreaction
    
    def __str__(self):
        return str(self.coreaction)


class Action:
    """actions taken on the group of items

    Raises:
        ValueError: The workload is too much for a day (implemented in _limitationsList)
    
    Returns
        Action (Union[Dict[CoreAction,int],None]) : The dictionary with
            - the keys corresponding to the index of the CoreAction
            - the values to the number of use of the associatedCoreAction
    """
    _limitationsList = [0.3,0.1]
    _limitations = dict()
    def __init__(self,action: Union[Dict[CoreAction,int],None]) -> None:
        self.ca_list = CoreAction.listCoreActions
        for i,a in enumerate(self.ca_list):
            self._limitations[str(a)] = self._limitationsList[i]
            
        self.action = action
        if action != None:
            return
        somme = 0
        for a,x in action.items():
            somme += self._limitations[a]*x
        if somme > 1:
            raise ValueError("Too much use")
    
    @staticmethod
    def fromDictInt(action: Union[Dict[int,int],None]) -> 'Action':
        """Generate an action from a dictionary of indexes (the order is defined by _list_actions in CoreAction)

        Args:
            action (Union[Dict[int,int],None]): The dictionary with
                - the keys corresponding to the index of the CoreAction
                - the values to the number of use of the associatedCoreAction

        Returns:
            Action: The corresponding action
        """
        if action == None:
            return Action(action)
        res = dict()
        for a,x in action.items():
            res[CoreAction.fromNumber(a)] = x
        return Action(res)
    
    @classmethod
    def fromListInt(self,action: Union[Dict[int,int],None]) -> 'Action':
        """Generate an action from a list (the order is defined by _list_actions in CoreAction)

        Args:
            action (Union[Dict[int,int],None]): The list with
                - the index corresponding to the index of the CoreAction
                - the values to the number of use of the associatedCoreAction

        Returns:
            Action: The corresponding action
        """
        self.ca_list = CoreAction.listCoreActions
        if action == None:
            return Action(action)
        res = dict()
        for i,x in enumerate(action):
            res[str(self.ca_list[i])] = x
        return Action(res)
    
    def listAction_aux(self,liste:List[int]) -> List['Action']:
        """Auxilary function (recursive) for listAction"""
        somme = 0
        for a,x in zip(self.ca_list,liste):
            somme += self._limitations[str(a)]*x
        if somme > 1:
            return []
        else:
            res = [Action.fromListInt(liste)]
            for i in range(len(liste)):
                new_liste = liste[:]
                new_liste[i] += 1
                res = res + self.listAction_aux(self,new_liste)
            return res

    @classmethod
    def listAction(self)-> List['Action']:
        """Generate all possible actions allowed by the _limitationsList

        Returns:
            List[Action]: List of all allowed actions
        """
        return self.listAction_aux(self,[0]*len(self.ca_list))

    def __str__(self):
        return str(self.action)

    def __repr__(self):
        return self.__str__()
    


class Observation:
    """Observation class

    Attributes:
        items_wear (List[Tuple[int, float]]): The wear indicator of each item (represented by their id).
    """

    def __init__(
        self,
        items: List[Item],
    ) -> None:

        self.items_wear = [(item.id, item.wear) for item in items]


class ObservationSpace(Space[Observation]):
    """ObservationSpace class"""

    def __init__(self) -> None:
        super().__init__()

    def sample(self) -> Observation:
        """
        Returns:
            Observation: a random observation
        """
        raise NotImplementedError

    def contains(self, x: Observation) -> bool:
        """
        Args:
            x (Observation): observation to check

        Returns:
            bool: True if the observation is in the observation space
        """
        raise NotImplementedError


class MainEnv(Env):
    """Gym-like enivronment."""

    def init(self, workers_nb: int, deployment_cost: float, items: List[Item]) -> None:
        super().__init__()

        self.workers_nb = workers_nb
        self.deployment_cost = deployment_cost

        self.items = items

        # Action and Observation spaces required by gym
        self.action_space = Discrete(self.max_n_clients_per_day)
        self.observation_space = ObservationSpace()

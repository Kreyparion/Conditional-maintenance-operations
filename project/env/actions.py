from typing import List, Dict, Tuple
from copy import deepcopy

class CoreAction:
    """unit actions on items, listed in _list_actions ("preventive","corrective")

    Raises:
        NameError: the action is not implemented
    """

    _list_actions = ["preventive", "corrective"]

    def __init__(self, action_type: str) -> None:
        if action_type in self._list_actions:
            self.coreaction = action_type
        else:
            raise NameError("Action not existing")

    @classmethod
    def fromNumber(self, i: int) -> "CoreAction":
        """create an object from an integer (the order is defined by _list_actions) : 0 for "preventive", 1 for"corrective"

        Args:
            i (int): the integer referencing the action

        Raises:
            IndexError: the index isn't implemented in the known action list

        Returns:
            CoreAction: The associated object
        """
        if i <= len(self._list_actions) and type(i)==int:
            return CoreAction(self._list_actions[i])
        else:
            raise IndexError("Index not corresponding to any Action")

    @classmethod
    def listCoreActions(self) -> List["CoreAction"]:
        """Generate the list of all Coreaction

        Returns:
            List[Action]: The list of all possible unit actions
        """
        return [CoreAction(x) for x in self._list_actions]

    def __eq__(self, other: "CoreAction"):
        if not isinstance(other, CoreAction):
            return NotImplemented
        return self.coreaction == other.coreaction

    def __repr__(self):
        return self.coreaction

    def __str__(self):
        return str(self.coreaction)

    def __hash__(self):
        return hash(self.coreaction)


class Action:
    """actions taken on the group of items

    Raises:
        ValueError: The workload is too much for a day (implemented in _limitationsList)

    Returns
        Action (Union[Dict[CoreAction,int],None]) : The dictionary with
            - the keys corresponding to the index of the CoreAction
            - the values to the number of use of the associatedCoreAction
    """

    _limitationsList = [0.1, 0.3]
    _ca_list = CoreAction.listCoreActions()
    
    def _init_limitations(_ca_list:List[CoreAction],_limitationsList:List[float]):
        limits = dict()
        if len(_limitationsList) != len(_ca_list):
            raise IndexError("not the same length between possible actions and limitations")
        for i, a in enumerate(_ca_list):
            limits[str(a)] = _limitationsList[i]
        return limits
    
    _limitations = _init_limitations(_ca_list,_limitationsList)
    
    def __init__(self, action: Dict[CoreAction, int]) -> None:
        self.action = action
        somme = 0.
        for a, x in action.items():
            somme += self._limitations[str(a)] * x
        if somme > 1:
            raise ValueError("Too much use")


    
    @staticmethod
    def fromDictInt(action: Dict[int, int]) -> "Action":
        """Generate an action from a dictionary of indexes (the order is defined by _list_actions in CoreAction)

        Args:
            action (Dict[int,int]): The dictionary with
                - the keys corresponding to the index of the CoreAction
                - the values to the number of use of the associatedCoreAction

        Returns:
            Action: The corresponding action
        """
        res = dict()
        for a, x in action.items():
            res[CoreAction.fromNumber(a)] = x
        return Action(res)

    @classmethod
    def fromListInt(self, action: List[int]) -> "Action":
        """Generate an action from a list (the order is defined by _list_actions in CoreAction)

        Args:
            action (Union[Dict[int,int],None]): The list with
                - the index corresponding to the index of the CoreAction
                - the values to the number of use of the associatedCoreAction

        Returns:
            Action: The corresponding action
        """
        res = dict()
        for i, x in enumerate(action):
            res[CoreAction(str(self._ca_list[i]))] = x
        return Action(res)

    @classmethod
    def ActionDoNothing(self):
        """Action with all Core Actions set to 0"""
        return self.fromListInt([0]*len(self._ca_list))

    @classmethod
    def _listAction_aux(self,nb_items:int) -> List[List[int]]:
        """Auxilary function (recursive) for listAction"""

        def valide(liste,nb_items:int):
            somme = 0
            nb_act = 0
            for a, x in zip(self._ca_list, liste):
                nb_act = nb_act + x
                if nb_act > nb_items:
                    return False
                somme += self._limitations[str(a)] * x
            if somme > 1:
                return False
            return True

        res = [[0 for i in range(len(self._ca_list))]]
        id = 0
        while id < len(self._ca_list):
            new_res = deepcopy(res)
            for x in new_res:
                a = x
                while 1:
                    a = deepcopy(a)
                    a[id] += 1
                    if not valide(a,nb_items):
                        break
                    res = res + [a]
            id += 1
        return res

    @classmethod
    def listAction(self,nb_items:int) -> List["Action"]:
        """Generate all possible actions allowed by the _limitationsList

        Returns:
            List[Action]: List of all allowed actions
        """
        return [self.fromListInt(x) for x in self._listAction_aux(nb_items)]

    def __eq__(self, other: "Action"):
        if not isinstance(other, Action):
            return NotImplemented
        return self.action == other.action
    
    def __str__(self):
        return str(self.action)

    def __repr__(self):
        return self.__str__()
    
    def to_tuple(self) -> Tuple[int]:
        return tuple(self.action.values())
    
    def __hash__(self) -> int:
        return hash(tuple(self.action.values()))

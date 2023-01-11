from itertools import product
from typing import Callable, List, Dict
from copy import deepcopy

from project.env.items import Item


class State:
    """Class that describes the state of the environment."""

    def __init__(self, continuous: bool, items: List[Item]) -> None:
        self.continuous = continuous
        self.items = items
        if items:
            self.wearing_func = items[0].wearing_func
            if not continuous:
                self.wear_ranges = [range(item.threshold + 1) for item in self.items]
                self.max_prods = [item.max_prod for item in self.items]

    @classmethod
    def get_states(
        self, thresholds: List[float], wearing_func: Callable, max_prods: List[float]
    ) -> List["State"]:
        """Method that returns all possible states in case the environment is discrete."""
        possibilities = [[0 for _ in range(len(thresholds))]]
        id = 0
        while id < len(thresholds):
            new_possibilities = deepcopy(possibilities)
            for x in new_possibilities:
                a = x
                while 1:
                    a = deepcopy(a)
                    if a[id] < thresholds[id]:
                        if id != 0:
                            if a[id]<a[id-1]:
                                a[id] += 1
                            else:
                                break
                        else:
                            a[id] += 1
                    else:
                        break
                    new_possibilities = new_possibilities + [a]
            id += 1
            possibilities = new_possibilities
        
        possible_states = [
            State(
                False,
                [
                    Item(
                        id=id,
                        max_prod=max_prod,
                        threshold=threshold,
                        wearing_func=wearing_func,
                        wear=wear,
                    )
                    for id, (max_prod, threshold, wear) in enumerate(
                        zip(
                            max_prods,
                            thresholds,
                            [possibilities[i][nb] for nb in range(len(thresholds))],
                        )
                    )
                ],
            )
            for i in range(len(possibilities))
        ]
        return possible_states
    
    def getList(self):
        return [item.wear/item.threshold for item in self.items]


    def wearing_step(self) -> None:
        for i in range(len(self.items)):
            self.items[i].wearing_step()
    
    @property
    def nb_items(self):
        return len(self.items)
    
    @staticmethod
    def from_lists(
        continuous: bool,
        max_prods: List[float],
        thresholds: List[float],
        wearing_func: Callable,
        wears: List[float] = None,
    ) -> "State":
        if wears is None:
            wears = [0] * len(max_prods)


        items = [
            Item(
                i,
                max_prod=max_prod,
                threshold=threshold,
                wearing_func=wearing_func,
                wear=wear,
            )
            for i, (max_prod, threshold, wear) in enumerate(
                zip(max_prods, thresholds, wears)
            )
        ]
        return State(continuous,items)

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.items == other.items
    
    def to_tuple(self):
        return tuple([item.wear for item in self.items])
    
    def __hash__(self) -> int:
        wears = [item.wear for item in self.items]
        return hash(tuple(wears))
    
    def __str__(self) -> str:
        return str(self.items)
    
    def __repr__(self) -> str:
        return self.__str__()
from itertools import product
from typing import Callable, List, Dict

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
        characteristics = []
        for threshold in thresholds:
            characteristics.append(range(threshold + 1))
            characteristics.append([True, False])
        possibilities = list(product(*characteristics))
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
                        is_nerfed=is_nerfed,
                    )
                    for id, (max_prod, threshold, wear, is_nerfed) in enumerate(
                        zip(
                            max_prods,
                            thresholds,
                            [possibilities[i][nb*2] for nb in range(len(thresholds))],
                            [possibilities[i][nb*2+1] for nb in range(len(thresholds))],
                        )
                    )
                ],
            )
            for i in range(len(possibilities))
        ]
        return possible_states

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
        nerfed_list: List[bool] = None,
    ) -> "State":
        if wears is None:
            wears = [0] * len(max_prods)
        if nerfed_list is None:
            nerfed_list = [False] * len(max_prods)

        items = [
            Item(
                i,
                max_prod=max_prod,
                threshold=threshold,
                wearing_func=wearing_func,
                wear=wear,
                is_nerfed=is_nerfed,
            )
            for i, (max_prod, threshold, wear, is_nerfed) in enumerate(
                zip(max_prods, thresholds, wears, nerfed_list)
            )
        ]
        return State(continuous,items)

    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.items == other.items
    
    def __hash__(self) -> int:
        wears = [item.threshold + 1 for item in self.items]
        nerfeds = [item.is_nerfed for item in self.items]
        return hash((tuple(wears), tuple(nerfeds)))
    
    def __str__(self) -> str:
        return str(self.items)
    
    def __repr__(self) -> str:
        return self.__str__()
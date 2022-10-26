from itertools import product
from project.env.items import Item
from typing import Callable, List, Dict


class State:
    """Class that describes the state of the environment."""

    def __init__(self, continuous: bool, items: List[Item]) -> None:
        self.continuous = continuous
        self.items = items
        self.wearing_func = items[0].wearing_func
        self.wear_ranges = [range(item.threshold) for item in self.items]
        self.max_prods = [item.max_prod for item in self.items]

    @classmethod
    def get_states(self) -> List["State"]:
        """Method that returns all possible states in case the environment is discrete."""
        if not self.continuous:
            return
        possibilites = product(*self.wear_ranges)
        possible_states = [
            State(
                [
                    Item(i, max_prod, threshold, self.wearing_func)
                    for i, (max_prod, threshold) in enumerate(
                        self.max_prods, wears_possible
                    )
                ]
            )
            for wears_possible in possibilites
        ]
        return possible_states

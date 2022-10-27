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
        self, max_prods: List[float], wear_ranges: List[range], wearing_func: Callable
    ) -> List["State"]:
        """Method that returns all possible states in case the environment is discrete."""
        possibilities = product(*wear_ranges)

        possible_states = [
            State(
                False,
                [
                    Item(i, max_prod, threshold, wearing_func)
                    for i, (max_prod, threshold) in enumerate(
                        zip(max_prods, wears_possible)
                    )
                ],
            )
            for wears_possible in possibilities
        ]
        return possible_states

    @classmethod
    def from_lists(
        continuous: bool,
        max_prods: List[float],
        thresholds: List[float],
        wearing_func: Callable,
        wears: List[float] = None,
    ) -> "State":
        if wears is None:
            items = [
                Item(
                    i, max_prod=max_prod, threshold=threshold, wearing_func=wearing_func
                )
                for i, (max_prod, threshold) in enumerate(zip(max_prods, thresholds))
            ]
        else:
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
        return items

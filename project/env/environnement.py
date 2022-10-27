from typing import List, Dict, Union, Tuple, Callable

from project.env.actions import Action
from project.env.states import State
from project.env.items import Item


class Environnement:
    def __init__(self, items: List[Item], continuous: bool = False) -> None:
        self.continuous = continuous
        self.items = items
        if not continuous:
            for item in self.items:
                item.threshold = int(item.threshold)

    def reset(self) -> State:
        pass

    def step(self, action: Action) -> Tuple[State, float, bool]:
        pass  # TODO@Etienne: Pour l'appliquer sur state

    def render(self) -> None:
        pass  # TODO@Théophile

    @classmethod
    def getPossibleActions(self, state: State) -> List[Action]:
        return Action.listAction()

    def getEveryState(self) -> List[State]:
        if self.continuous:
            raise RuntimeError("Runtime set to continuous")
        max_prods = [item.max_prod for item in self.items]
        thresholds = [item.threshold for item in self.items]
        wearing_func = self.items[0].wearing_func
        return State.get_states(
            max_prods=max_prods, thresholds=thresholds, wearing_func=wearing_func
        )

    @classmethod
    def from_list(
        continuous: bool,
        max_prods: List[float],
        thresholds: List[float],
        wearing_func: Callable,
    ) -> "Environnement":
        items = [
            Item(
                i,
                max_prod=max_prod,
                threshold=threshold,
                wearing_func=wearing_func,
            )
            for i, (max_prod, threshold) in enumerate(zip(max_prods, thresholds))
        ]
        return Environnement(items, continuous)

    @classmethod
    def from_floats(
        continuous: bool,
        nb_items: int,
        max_prod: float,
        threshold: float,
        wearing_func: Callable,
    ) -> "Environnement":
        max_prods = nb_items * [max_prod]
        thresholds = nb_items * [threshold]
        return Environnement.from_list(continuous, max_prods, thresholds, wearing_func)

from typing import List, Dict, Union, Tuple, Callable

from project.env.actions import Action
from project.env.states import State
from project.env.items import Item
from project.env.executions import execution


class Environnement:
    def __init__(self, items: List[Item], continuous: bool = False) -> None:
        self.continuous = continuous
        self.items = items
        if not continuous:
            for item in self.items:
                item.threshold = int(item.threshold)

        self._initial_wears, self._initial_nerfs = [], []

        for item in self.items:
            self._initial_wears.append(item.wear)
            self._initial_nerfs.append(item.is_nerfed)

    def reset(self) -> State:
        for item, init_wear, init_nerf in zip(
            self.items, self._initial_wears, self._initial_nerfs
        ):
            item.wear = init_wear
            item.is_nerfed = init_nerf
        return State(self.continuous, self.items)

    def step(self, action: Action) -> Tuple[State, float, bool]:
        pass  # TODO@Paul: Pour l'appliquer sur state

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

    @staticmethod
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
        self,
        continuous: bool,
        nb_items: int,
        max_prod: float,
        threshold: float,
        wearing_func: Callable,
    ) -> "Environnement":
        max_prods = nb_items * [max_prod]
        thresholds = nb_items * [threshold]
        return self.from_list(continuous, max_prods, thresholds, wearing_func)

    @classmethod
    def init(self, execution_type: str):
        execution_properties = execution(execution_type)
        return self.from_list(
            execution_properties[0],
            execution_properties[1],
            execution_properties[2],
            execution_properties[3],
        )

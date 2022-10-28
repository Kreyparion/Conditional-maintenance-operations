from typing import List, Dict, Union, Tuple, Callable

from project.env.actions import Action
from project.env.states import State
from project.env.items import Item
from project.env.executions import execution


class Environnement:
    def __init__(
        self,
        items: List[Item],
        prev_efficiency: float,
        repair_thrs: float,
        ship_cost: float = 4,
        corr_cost: float = 4,
        prev_cost: float = 1,
        continuous: bool = False,
    ) -> None:
        self.continuous = continuous
        self.prev_efficiency = prev_efficiency
        self.repair_thrs = repair_thrs
        self.ship_cost = ship_cost
        self.corr_cost = corr_cost
        self.prev_cost = prev_cost

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
        action_dict = action.action
        nb_cor = action_dict["corrective"] if "corrective" in action_dict else 0
        nb_pre = action_dict["preventive"] if "preventive" in action_dict else 0
        nb_nerf = action_dict["nerf"] if "nerf" in action_dict else 0
        self.indexes = []
        for i, item in enumerate(self.items):
            self.indexes.append((i, item.wear, item.is_nerfed))

        self.indexes.sort(
            key=lambda x: -x[1]
        )  # The first indexes are those of the most wore items
        cor_act_used, pre_act_used = 0, 0
        for index_tuple in self.indexes:
            item_index = index_tuple[0]
            if (
                cor_act_used < nb_cor
                and index_tuple[1] == self.items[item_index].threshold
            ):  # The item is shut down and we can fix it
                self.items[item_index].wear = self.repair_thrs
                cor_act_used += 1
            elif (
                pre_act_used < nb_pre
                and index_tuple[1] != self.items[item_index].threshold
            ):
                wear = self.items[item_index].wear
                self.items[item_index].wear = max(
                    self.repair_thrs, wear - self.prev_efficiency
                )
                pre_act_used += 1

        self.indexes.sort(key=lambda x: (x[2], -x[1]))
        nerf_used = 0
        for index_tuple in self.indexes:
            if nerf_used >= nb_nerf:
                break
            self.items[index_tuple[0]].is_nerfed = True
            nerf_used += 1

        return State(self.continuous, self.items), self.reward(nb_cor, nb_pre), False

    def reward(self, nb_corrective, nb_preventif):
        rew = 0
        for item in self.items:
            rew += item.productivity
        rew -= self.ship_cost
        rew -= nb_corrective * self.corr_cost
        rew -= nb_preventif * self.prev_cost
        return rew

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

from os import stat
from typing import List, Dict, Union, Tuple, Callable

from project.env.actions import Action, CoreAction
from project.env.states import State
from project.env.items import Item
from project.env.executions import execution

import random
import matplotlib.pyplot as plt
import numpy as np
import wandb
from copy import deepcopy

class Environnement:
    def __init__(
        self,
        items: List[Item],
        continuous: bool = False,
        prev_efficiency: float=2,
        repair_thrs: float = 0,
        ship_cost: float = 54,
        corr_cost: float = 54,
        prev_cost: float = 18,
    ) -> None:
        self.continuous = continuous
        self.prev_efficiency = prev_efficiency
        self.repair_thrs = repair_thrs
        self.ship_cost = ship_cost
        self.corr_cost = corr_cost
        self.prev_cost = prev_cost
        
        
        self.step_number = 0
        self.last_reward = 0
        self.last_500_rewards = []
        self.line1 = []
        self.fig = None
        wandb.init(project="proj_rl")
        wandb.config = {
            "number_of_items": len(items),
            "continuous": continuous,
            "prev_efficiency": prev_efficiency,
            "repair_thrs": repair_thrs,
            "ship_cost": ship_cost,
            "corr_cost": corr_cost,
            "prev_cost": prev_cost,
        }
        self.items = items
        self.state = None
        
        if not continuous:
            for item in self.items:
                item.threshold = int(item.threshold)

        self._initial_wears = []

        for item in self.items:
            self._initial_wears.append(item.wear)


    def reset(self) -> State:
        
        item_wear = []
        for item, init_wear in zip(
            self.items, self._initial_wears
        ):
            item_wear.append(random.randint(0, item.threshold))
        item_wear.sort(reverse=True)
        for i,item in enumerate(self.items):
            item.wear = item_wear[i]

        self.state = State(self.continuous, self.items)
        return self.state
    
    def initial_state(self) -> State:
        initial_item = deepcopy(self.items)
        for item, init_wear in zip(
            initial_item, self._initial_wears
        ):
            item.wear = init_wear
        return State(self.continuous, initial_item)

    def step(self, action: Action) -> Tuple[State, float, bool]:
        action_dict = action.action
        nb_cor = action_dict[CoreAction("corrective")]
        nb_pre = action_dict[CoreAction("preventive")]
        self.items = self.state.items
        for item in self.items:
            item.wearing_step()
        self.indexes = []
        for i, item in enumerate(self.items):
            self.indexes.append((i, item.wear))

        self.indexes.sort(
            key=lambda x: -x[1]
        )  # The first indexes are those of the most wore items
        cor_act_used, pre_act_used = 0, 0
        for index_tuple in self.indexes:
            item_index = index_tuple[0]
            if (
                cor_act_used < nb_cor
                and self.items[item_index].wear == self.items[item_index].threshold
            ):  # The item is shut down and we can fix it
                self.items[item_index].wear = self.repair_thrs
                cor_act_used += 1
            elif (
                pre_act_used < nb_pre
                and self.items[item_index].wear != self.items[item_index].threshold
                and self.items[item_index].wear != 0
            ):
                wear = self.items[item_index].wear
                self.items[item_index].wear = max(self.repair_thrs, wear - self.prev_efficiency)
                pre_act_used += 1


        self.items.sort(key=lambda x: -x.wear)
        self.step_number += 1
        self.state = State(self.continuous, self.items)
        self.last_reward = self.reward(nb_cor, nb_pre)
        done = False
        #if self.step_number % 5000 == 0:
        #    done = True
        return self.state, self.reward(nb_cor, nb_pre), done

    def reward(self, nb_corrective, nb_preventif):
        rew = 0
        for item in self.items:
            rew += item.productivity
        if nb_corrective+nb_preventif>0:
            rew -= self.ship_cost
        rew -= nb_corrective * self.corr_cost
        rew -= nb_preventif * self.prev_cost
        return rew

    
    def render(self) -> None:
        self.last_500_rewards.append(self.last_reward)
        if len(self.last_500_rewards) > 500:
            self.last_500_rewards.pop(0)
        if self.step_number % 500 == 0:
            wandb.log({"reward_500": np.mean(self.last_500_rewards)})
        
    def getPossibleActions(self, state: State) -> List[Action]:
        return Action.listAction(state.nb_items)

    def getEveryState(self) -> List[State]:
        if self.continuous:
            raise RuntimeError("Runtime set to continuous")
        max_prods = [item.max_prod for item in self.items]
        thresholds = [item.threshold for item in self.items]
        wearing_func = self.items[0].wearing_func
        return State.get_states(
            max_prods=max_prods, thresholds=thresholds, wearing_func=wearing_func
        )
    
    def out_of_order_state(self) -> State:
        self.ooo_items = deepcopy(self.items)
        for item in self.ooo_items:
            item.wear = item.threshold
        return State(self.continuous, self.ooo_items)

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

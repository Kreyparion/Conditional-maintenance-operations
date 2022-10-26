from project.env.actions import Action
from project.env.states import State
from typing import List, Dict, Union, Tuple


class Environnement:
    def __init__(self, continuous: bool = False) -> None:
        self.continuous = continuous

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
        return State.get_states()

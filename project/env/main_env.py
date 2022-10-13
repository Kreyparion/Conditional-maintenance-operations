from enum import Enum
from typing import List

from gym import Env
from gym.spaces import Discrete, Space

from project.env.items import Item


class ActionType(Enum):
    PREVENTIVE = ("preventive",)
    CORRECTIVE = "cprrective"


class Observation:
    """Observation class

    Attributes:
        items_wear (List[Tuple[int, float]]): The wear indicator of each item (represented by their id).
    """

    def __init__(
        self,
        items: List[Item],
    ) -> None:

        self.items_wear = [(item.id, item.wear) for item in items]


class ObservationSpace(Space[Observation]):
    """ObservationSpace class"""

    def __init__(self) -> None:
        super().__init__()

    def sample(self) -> Observation:
        """
        Returns:
            Observation: a random observation
        """
        raise NotImplementedError

    def contains(self, x: Observation) -> bool:
        """
        Args:
            x (Observation): observation to check

        Returns:
            bool: True if the observation is in the observation space
        """
        raise NotImplementedError


class MainEnv(Env):
    """Gym-like enivronment."""

    def init(self, workers_nb: int, deployment_cost: float, items: List[Item]) -> None:
        super().__init__()

        self.workers_nb = workers_nb
        self.deployment_cost = deployment_cost

        self.items = items

        # Action and Observation spaces required by gym
        self.action_space = Discrete(self.max_n_clients_per_day)
        self.observation_space = ObservationSpace()

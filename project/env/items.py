from abc import ABC, abstractmethod
from typing import Callable


class ABCItem(ABC):
    """Class that represents a single unit item (eg windmill)."""

    def __init__(
        self,
        id: int,
        wear: float = 0,
        nerf_value: float = 0.75,
        is_nerfed: bool = False,
    ) -> None:
        self.id = id
        self.wear = wear
        self.nerf_value = nerf_value
        self.is_nerfed = is_nerfed

    @abstractmethod
    def wearing_step(self) -> None:
        """Function that update the wear attribute according to the item behaviour."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def productivity(self) -> float:
        """Returns the energy produced during a day."""
        raise NotImplementedError


class Item(ABCItem):
    """Class that represents an item whose wearing steps following a continuous process.

    Args:
        id (int): The item's id.
        wearing_func (Callable): The function returning wearing steps.
        threshold (float): If the wear of the item is greater than the threshold, it is stopped.
        max_prod (float): The energy produced by the item at full capacity.
    """

    def __init__(
        self,
        id: int,
        threshold: float,
        wearing_func: Callable,
        wear: float = 0,
        nerf: float = 0.75,
        max_prod: float = 1.,
        is_nerfed: bool = False,
    ) -> None:
        super().__init__(id, wear, nerf, is_nerfed)
        self.threshold = threshold
        self.wearing_func = wearing_func
        self.max_prod = max_prod

    def wearing_step(self) -> None:
        if self.wear < self.threshold:
            self.wear = min(
                self.threshold,
                self.wearing_func(self.threshold,self.wear),
            )

    @property
    def productivity(self) -> float:
        if self.is_nerfed:
            return (
                self.nerf_value * self.max_prod if self.wear < self.threshold else 0.0
            )
        return self.max_prod if self.wear < self.threshold else 0.0

    def reset(self) -> None:
        self.wear = 0
        self.is_nerfed = False
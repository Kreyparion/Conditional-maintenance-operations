from abc import ABC, abstractmethod
from typing import Callable


class ABCItem(ABC):
    """Class that represents a single unit item (eg windmill)."""

    def __init__(self, id: int) -> None:
        self.id = id
        self.wear = 0

    @abstractmethod
    def wearing_step(self) -> None:
        """Function that update the wear attribute according to the item behaviour."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    @property
    def productivity(self) -> float:
        """Returns the energy produced during a day."""
        raise NotImplementedError


class Item(ABCItem):
    """Class that represents an item whose wearing steps following a continuous process.

    Args:
        id (int): The item's id.
        max_prod (float): The energy produced by the item at full capacity.
        wearing_func (Callable): The function returning wearing steps.
        threshold (float): If the wear of the item is greater than the threshold, it is stopped.
    """

    def __init__(
        self, id: int, max_prod: float, threshold: float, wearing_func: Callable
    ) -> None:
        super().__init__(id)
        self.threshold = threshold
        self.wearing_func = wearing_func
        self.max_prod = max_prod

    def wearing_step(self) -> None:
        if self.wear < self.threshold:
            self.wear = min(
                self.threshold,
                self.wear + self.wearing_func(),
            )

    @property
    def productivity(self) -> float:
        return self.max_prod if self.wear < self.threshold else 0.0

    def reset(self) -> None:
        self.wear = 0
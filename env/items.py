from abc import ABC, abstractmethod

import numpy as np


class Item(ABC):
    """Class that represents a single unit item (eg windmill)."""

    def __init__(self, id: int) -> None:
        self.id = id
        self.wear = 0

    @abstractmethod
    def wearing_func(self) -> None:
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


class GammaItem(Item):
    """Class that represents an item whose wearing steps follow a Gamma process.

    Args:
        id (int): The item's id.
        max_prod (float): The energy produced by the item at full capacity.
        threshold (float): If the wear of the item is greater than the threshold, it is stopped.
        shape (float): The shape of the gamma distribution. Must be non-negative.
        scale (float): The scale of the gamma distribution. Must be non-negative. Default is equal to 1.
    """

    def __init__(
        self, id: int, max_prod: float, threshold: float, shape: float, scale: float = 1
    ) -> None:
        super().__init__(id)
        self.threshold = threshold
        self.shape = shape
        self.scale = scale
        self.max_prod = max_prod

    def wearing_func(self) -> None:
        if self.wear < self.threshold:
            self.wear = max(
                self.threshold,
                self.wear + np.random.gamma(sahpe=self.shape, scale=self.scale),
            )

    @property
    def productivity(self) -> float:
        return self.max_prod if self.wear < self.threshold else 0.0

    def reset(self) -> None:
        self.wear = 0

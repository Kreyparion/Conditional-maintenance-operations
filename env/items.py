from abc import ABC, abstractmethod

import numpy as np


class Item(ABC):
    """Class that represent a single unit item (eg windmill)"""

    def __init__(self, id: int) -> None:
        self.id = id
        self.wear = 0

    @abstractmethod
    def wearing_func(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    @property
    def productivity(self) -> float:
        raise NotImplementedError


class GammaCWindmill(Item):
    def __init__(
        self, id: int, max_prod: float, threshold: float, shape: float, scale: float
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
        return self.max_prod

    def reset(self) -> None:
        self.wear = 0

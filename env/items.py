from abc import ABC, abstractmethod


class Item(ABC):
    """Class that represent a single unit item (eg windmill)"""

    def __init__(self, id: int) -> None:
        self.id = id

    @abstractmethod
    def wearing_func(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    @property
    def wear(self) -> float:
        raise NotImplementedError

    @abstractmethod
    @property
    def productivity(self) -> float:
        raise NotImplementedError

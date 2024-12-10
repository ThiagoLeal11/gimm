from abc import ABC, abstractmethod


class Tween(ABC):
    def __init__(self, start: int, duration: int):
        self.start = start
        self.end = start + duration
        self.duration = duration

    @abstractmethod
    def compute(self, t: int) -> float:
        """
        Compute the tween value for a given time. Returns a value between 0 and 1.
        It is guaranteed that the time t is between start and end.
        """
        pass

    def __call__(self, t: int) -> float:
        if t < self.start:
            return 0.0

        if t >= self.end:
            return 1.0

        return self.compute(t)


class LinearTween(Tween):
    def compute(self, t: int) -> float:
        return self.start + (t / self.duration)

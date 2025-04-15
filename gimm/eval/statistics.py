from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class StatisticAggregator(ABC):
    @abstractmethod
    def update(self, value: Tensor) -> StatisticAggregator:
        pass

    @abstractmethod
    def compute(self) -> Tensor:
        pass

    @abstractmethod
    def reset(self) -> StatisticAggregator:
        pass


# TODO: Make tests for this clas
class MeanMetric(StatisticAggregator):
    """A class for computing running mean of metrics incrementally.

    This class maintains a running sum and count to compute the mean of values
    that are processed in batches, making it memory efficient for large datasets.

    Attributes:
        sum (Tensor): Running sum of all values
        count (Tensor): Total number of values processed for each position
        axis (Optional[int]): Axis along which to compute mean (None for global mean)
    """

    def __init__(self, axis: Optional[int] = None, device: str | torch.device | None = None):
        self.axis = axis
        self.device = device
        self.sum = self.count = None
        self.reset()

    def _agg(self, update_sum: Tensor, update_count: Tensor):
        if self.sum is None or self.count is None:
            self.sum = update_sum
            self.count = update_count
            return

        if self.axis is not None and self.axis > 0:
            self.sum = torch.cat([self.sum, update_sum], dim=0)
            self.count = torch.cat([self.count, update_count], dim=0)
            return

        self.sum += update_sum
        self.count += update_count

    def reset(self) -> MeanMetric:
        self.sum = self.count = None
        return self

    def update(self, batch: Tensor) -> MeanMetric:
        if not isinstance(batch, Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        if batch.device != self.device:
            batch = batch.to(self.device)

        if self.axis is None:
            self._agg(
                batch.sum(),
                torch.tensor(batch.numel(), device=batch.device)
            )
            return self

        dim = self.axis if self.axis >= 0 else batch.ndim + self.axis

        if not -batch.ndim <= dim < batch.ndim:
            raise ValueError(f"Axis {self.axis} is out of bounds for tensor of dimension {batch.ndim}")

        self._agg(
            batch.sum(dim=dim),
            torch.full_like(batch, 1).sum(dim=dim)
        )

        return self

    def compute(self) -> Tensor:
        if self.sum is None or self.count is None:
            raise ZeroDivisionError("No values have been added yet")

        return self.sum / self.count


# TODO: Make tests for this class
class CovarianceMetric(StatisticAggregator):
    """ A class for computing running covariance of metrics incrementally.

    Iteratively compute covariance matrix similar to np.cov.
    Allows online computation without storing all data points.

    Attributes:
        n (int): Number of samples processed
        mean (Tensor): Running mean of all values
        M2 (Tensor): Sum of squares of differences from the current mean
        ddof (int): Delta degrees of freedom for covariance calculation
    """

    def __init__(self, ddof: int = 1):
        """
        Initialize the covariance calculator.

        Args:
            ddof: Delta degrees of freedom. Default is 1 for sample covariance.
        """
        self.n = self.mean = self.M2 = None
        self.ddof = ddof
        self.reset()

    def reset(self) -> CovarianceMetric:
        """Reset all statistics."""
        self.n = 0
        self.mean = None
        self.M2 = None
        return self

    def update(self, batch: Tensor) -> CovarianceMetric:
        """
        Update statistics with new batch of data.
        """
        # Ensure input is 2D
        if batch.dim() == 1:
            batch = batch.view(-1, 1)

        batch_size, num_features = batch.shape

        # Initialize mean and M2 if this is first batch
        if self.mean is None:
            self.mean = torch.zeros(num_features, dtype=batch.dtype, device=batch.device)
            self.M2 = torch.zeros(num_features, num_features, dtype=batch.dtype, device=batch.device)

        # Welford's online algorithm adapted for covariance
        for i in range(batch_size):
            self.n += 1
            delta = batch[i] - self.mean
            self.mean += delta / self.n
            delta2 = batch[i] - self.mean
            self.M2 += torch.outer(delta, delta2)

        return self

    def compute(self) -> Tensor:
        """
        Compute the current covariance matrix.

        Returns:
            Covariance matrix as a tensor of shape (num_features, num_features)
        """
        if self.n < (1 + self.ddof):
            raise ValueError(f"Need at least {1 + self.ddof} samples to compute covariance")
        return self.M2 / (self.n - self.ddof)

from unittest import TestCase

import unittest

import numpy as np
import torch

from gimm.eval.statistics import MeanMetric, CovarianceMetric


class TestMeanMetric(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""

    def test_initialization(self):
        """Test proper initialization of the metric."""
        mean = MeanMetric()
        with self.assertRaises(ZeroDivisionError):
            mean.compute()

    def test_reset(self):
        """Test reset functionality."""
        mean = MeanMetric()
        mean.update(torch.tensor([1.0, 2.0]))
        mean.compute()

        mean.reset()
        with self.assertRaises(ZeroDivisionError):
            mean.compute()

    def test_global_mean(self):
        """Test basic mean computation without axis specification."""
        # Single update
        mean = MeanMetric()
        mean.update(torch.tensor([1.0, 3.0]))
        self.assertEqual(mean.compute().item(), 2.0)

        # Multiple updates
        mean.update(torch.tensor([2.0, 4.0]))
        self.assertEqual(mean.compute().item(), 2.5)

        # 2D tensor
        mean.reset()
        mean.update(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        self.assertEqual(mean.compute().item(), 6.5)

    def test_2d_axis_mean(self):
        """Test mean computation along different axes."""
        tensor = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]])
        mean0 = MeanMetric(axis=0)
        mean0.update(tensor[0:2]).update(tensor[2:5])
        expected = torch.tensor([5., 6.])
        self.assertTrue(torch.allclose(mean0.compute(), expected))

        mean1 = MeanMetric(axis=1)
        mean1.update(tensor[0:3]).update(tensor[3:5])
        expected = torch.tensor([1.5, 3.5, 5.5, 7.5, 9.5])
        self.assertTrue(torch.allclose(mean1.compute(), expected))

    def test_4d_axis_mean(self):
        """Test mean computation along different axes."""
        # Batch of 6 images of size 5x4 with 3 channels
        imgs = torch.range(0, 359).view([6, 5, 4, 3])

        for axis in range(4):
            mean = MeanMetric(axis=axis)
            mean.update(imgs[0:2]).update(imgs[2:5]).update(imgs[5:6])
            result = mean.compute()
            expected = imgs.mean(dim=axis)
            self.assertTrue(torch.allclose(result, expected))

    def test_error_handling(self):
        """Test error cases and edge conditions."""
        # Test computing before any updates
        mean = MeanMetric()
        with self.assertRaises(ZeroDivisionError):
            mean.compute()

        # Test invalid input type
        with self.assertRaises(ValueError):
            mean.update([1, 2, 3])  # Not a tensor

        # Test invalid axis
        mean = MeanMetric(axis=2)
        with self.assertRaises(ValueError):
            mean.update(torch.tensor([1, 2, 3]))


class TestCovarianceMetric(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""

    def test_initialization(self):
        """Test proper initialization of the metric."""
        cov = CovarianceMetric()
        with self.assertRaises(ValueError):
            cov.compute()

    def test_reset(self):
        """Test reset functionality."""
        cov = CovarianceMetric()
        cov.update(torch.tensor([1.0, 2.0]))
        cov.compute()

        cov.reset()
        with self.assertRaises(ValueError):
            cov.compute()

    def test_global_covariance(self):
        """Test basic covariance computation without axis specification."""
        # Single update
        cov = CovarianceMetric()

        t1 = torch.tensor([1.0, 3.0])
        cov.update(t1)
        self.assertEqual(cov.compute().item(), np.cov(t1.numpy(), rowvar=False))

        # Multiple updates
        t2 = torch.tensor([2.0, 4.0])
        cov.update(t2)
        self.assertEqual(cov.compute().item(), np.cov(np.concatenate([t1.numpy(), t2.numpy()]), rowvar=False).item())

        # 2D tensor
        cov.reset()
        t3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        cov.update(t3)
        self.assertEqual(cov.compute().item(), np.cov(t3.numpy(), rowvar=False))

    def test_2d_axis_covariance(self):
        """Test covariance computation along different axes."""
        imgs = torch.range(0, 359).view([6, 5*4*3])

        for axis in range(4):
            mean = MeanMetric(axis=axis)
            mean.update(imgs[0:2]).update(imgs[2:5]).update(imgs[5:6])
            result = mean.compute()
            expected = np.cov(imgs.numpy(), rowvar=False)
            self.assertTrue(torch.allclose(result, torch.tensor(expected, dtype=torch.float32)))


def cov(m, rowvar=True, ddof=None):
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")

    # Handles complex arrays too
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    X = np.array(m, ndmin=2)
    # if not rowvar and X.shape[0] != 1:
    X = X.T

    # if X.shape[0] == 0:
    #     return np.array([]).reshape(0, 0)

    if ddof is None:
        ddof = 0

    avg = np.average(X, axis=1)

    # Determine the normalization
    fact = X.shape[1] - ddof

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= avg[:, None]
    X_T = X.T
    c = np.dot(X, X_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()
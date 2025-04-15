import json

import numpy as np
import scipy
import torch
from torch import Tensor
from torchvision import transforms

from gimm.eval.compute import EvalMetric
from gimm.eval.models.inception_v3 import InceptionV3
from reference.fid_score import altered_main


class FrechetInceptionDistance(EvalMetric):
    def __init__(self, samples: int, device: torch.device = None):
        super().__init__(samples, device)

        self.real_activations = []
        self.fake_activations = []

        self.model = InceptionV3(resize_input=True)
        self.model.to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(
                size=(299, 299),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            )
        ])

    def reset_real_distribution(self) -> None:
        self.real_activations = []
        self.real_dist = {}

    def reset_fake_distribution(self) -> None:
        self.fake_activations = []

    def update(self, batch: tuple[Tensor, Tensor]) -> None:
        imgs, labels = batch
        real_imgs, fake_imgs = imgs[labels == 1], imgs[labels == 0]

        if self.should_compute_real_distribution() and real_imgs.size(0) > 0:
            activations = self._get_activations(real_imgs)
            self.real_activations.append(activations.cpu().numpy())

        activations = self._get_activations(fake_imgs)
        self.fake_activations.append(activations.cpu().numpy())

    def compute(self) -> dict[str, any]:
        self.real_activations = np.concatenate(self.real_activations, axis=0)
        # self.fake_activations = np.concatenate(self.fake_activations, axis=0)

        real_mu = np.mean(self.real_activations, axis=0)
        real_sigma = np.cov(self.real_activations, rowvar=False)

        # fake_mu = np.mean(self.fake_activations, axis=0)
        # fake_sigma = np.cov(self.fake_activations, rowvar=False)

        self.real_dist = {
            "mu": real_mu,
            "sigma": real_sigma
        }

        ground_mu, ground_sigma = altered_main()
        # # Compared the two numpy arrays
        # diff = {
        #     "mu": np.linalg.norm(real_mu - ground_mu),
        #     "sigma": np.linalg.norm(real_sigma - ground_sigma)
        # }
        # print(json.dumps({k: v.tolist() for k, v in diff.items()}))
        # raise ValueError("Stop All")

        # return calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        return calculate_frechet_distance(real_mu, real_sigma, ground_mu, ground_sigma)

    def _get_activations(self, imgs: Tensor) -> Tensor:
        scaled_imgs = self.transform(imgs).to(self.device)
        with torch.no_grad():
            pred = self.model(scaled_imgs)[0]
        return pred.squeeze(3).squeeze(2)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    # TODO: try to speed things up with https://github.com/w86763777/pytorch-image-generation-metrics/tree/master

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
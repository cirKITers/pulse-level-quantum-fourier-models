"""Small shared pieces: how a target is drawn, and how a fit is scored."""

import numpy as np
from scipy.stats import anderson_ksamp, energy_distance, wasserstein_distance


class Sampling:
    @staticmethod
    def uniform_circle(rng, low=0.0, high=1.0, size=None, density_correct=True):
        """Random number generator for complex numbers sampled inside the unit circle

        Args:
            low (float, optional): Minimum Radius. Defaults to 0.0.
            high (float, optional): Maximum Radius. Defaults to 1.0.
            size (int, optional): Number of samples. Defaults to None.
        """

        return np.sqrt(rng.uniform(low, high, size)) * np.exp(
            2j * np.pi * rng.uniform(low=0, high=1, size=size)
        )


class Losses:
    @staticmethod
    def mse(prediction, target):
        return np.mean((prediction - target) ** 2)

    @staticmethod
    def null_loss(prediction, target):
        return 0.0

    @staticmethod
    def kl_divergence(prediction, target):
        var_pred = prediction.var()
        var_target = target.var()
        mean_pred = prediction.mean()
        mean_target = target.mean()

        return 0.5 * np.sum(
            np.log(var_target / var_pred)
            + (var_pred + (mean_pred - mean_target) ** 2) / var_target
            - 1
        )

    @staticmethod
    def huber_loss(prediction, target, delta=1.0):
        a = prediction - target
        abs_a = np.abs(a)
        return np.mean(
            np.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))
        )

    @staticmethod
    def wasserstein_distance(prediction, target):
        return wasserstein_distance(prediction, target)

    @staticmethod
    def anderson_ksamp(prediction, target):
        return anderson_ksamp([prediction, target]).statistic

    @staticmethod
    def energy_distance(prediction, target):
        return energy_distance(prediction, target)

    @staticmethod
    def fmse(prediction, target):
        return np.mean(np.abs(prediction - target))

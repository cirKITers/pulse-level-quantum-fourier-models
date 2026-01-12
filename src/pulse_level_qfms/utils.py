import pennylane as qml
import pennylane.numpy as np
import plotly.graph_objects as go

import torch
from scipy.stats import wasserstein_distance, anderson_ksamp, energy_distance
from os.path import isfile, basename
from typing import Any, Dict, List
from kedro.io import AbstractDataset
import mlflow


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
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            return torch.mean((prediction - target) ** 2)
        else:
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

        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            return 0.5 * torch.sum(
                torch.log(var_target / var_pred)
                + (var_pred + (mean_pred - mean_target) ** 2) / var_target
                - 1
            )
        else:
            return 0.5 * np.sum(
                np.log(var_target / var_pred)
                + (var_pred + (mean_pred - mean_target) ** 2) / var_target
                - 1
            )

    @staticmethod
    def huber_loss(prediction, target, delta=1.0):
        a = prediction - target
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            # return torch.nn.functional.huber_loss(prediction, target)
            abs_a = torch.abs(a)
            return torch.mean(
                torch.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))
            )
        else:
            abs_a = np.abs(a)
            return np.mean(
                np.where(abs_a <= delta, 0.5 * a**2, delta * (abs_a - 0.5 * delta))
            )

    @staticmethod
    def wasserstein_distance(prediction, target):
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            target = target.detach().numpy()
            prediction = prediction.detach().numpy()
        return wasserstein_distance(prediction, target)

    @staticmethod
    def anderson_ksamp(prediction, target):
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            target = target.detach().numpy()
            prediction = prediction.detach().numpy()
        return anderson_ksamp([prediction, target]).statistic

    @staticmethod
    def energy_distance(prediction, target):
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            target = target.detach().numpy()
            prediction = prediction.detach().numpy()
        return energy_distance(prediction, target)

    @staticmethod
    def fmse(prediction, target):
        if type(target) == torch.Tensor and type(prediction) == torch.Tensor:
            target = target.detach().numpy()
            prediction = prediction.detach().numpy()
        return np.mean(np.abs(prediction - target))


class MlFlowPlotlyArtifact(AbstractDataset):
    """
    This class provides a central point for reporting figures via MlFlow instead of writing them via Kedro.
    Idea is, that kedro still handles the figure data and reporting takes form of individual catalog entries.
    This way the kedro "spirit" is preserved while using MlFlow for experiment tracking.
    """

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            filename=self._filename,
            load_args=self._load_args,
            save_args=self._save_args,
        )

    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        self._filepath = filepath
        self._filename = basename(filepath)
        default_save_args = {}
        default_load_args = {}

        self._load_args = (
            {**default_load_args, **load_args}
            if load_args is not None
            else default_load_args
        )
        self._save_args = (
            {**default_save_args, **save_args}
            if save_args is not None
            else default_save_args
        )

    def _load(self):
        raise NotImplementedError

    def _save(self, fig) -> None:
        mlflow.log_figure(fig, self._filename)

    def _exists(self) -> bool:
        return isfile(self._filepath)


def create_time_domain_viz(model, train_loader, noise_params):
    x = train_loader.dataset.tensors[0].numpy().squeeze()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=train_loader.dataset.tensors[1].numpy(),
            mode="lines+markers",
            name="target",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=model(
                params=model.params,
                inputs=x,
                noise_params=noise_params,
                execution_type="expval",
                force_mean=True,
            ),
            mode="lines+markers",
            name="prediction",
        )
    )
    fig.update_layout(
        title="Time domain visualization",
        xaxis_title="Time",
        yaxis_title="Amplitude",
        template="plotly_white",
    )

    return fig

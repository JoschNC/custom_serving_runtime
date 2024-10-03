# Imports
from typing import List
from numpy import asarray, array, median
from pandas import Series
from sklearn.preprocessing import normalize
import time
from datetime import timedelta
from src.ai_models.abstract_model import AbstractModel, ModelCategory
from math import e, isnan
from dataclasses import dataclass


def exp_decay(t, n_zero=1, halflife=10, reverse=False):
    t = -t if reverse else t
    return n_zero * 2 ** (-t / halflife)


def lin_decay(t, n_zero=1, steps=10, reverse=False):
    if reverse:
        t = -t
        if t >= -steps:
            return 0
    else:
        if t >= steps:
            return 0
    return n_zero * 1 - (t / steps)


# Model code
@dataclass
class KernelRegression(AbstractModel):
    """Kernel regression with radial basis kernel."""
    name: str 
    superroute: str
    category: ModelCategory = ModelCategory.SUPERROUTE
    bandwidth: float = 8.0
    recursive: bool = True
    recursive_switch: int = 0
    lookback_type: str = "original"
    lookback_cutoff: int = 0
    decay_rate: int = 3
    train_size: int = 100
    test_size: int = 100
    buffer_size: int = 10

    def __post_init__(self):
        super().__init__(name = self.name, category = self.category, superroute = self.superroute)
        # Data
        self.set_metrics()
        self.bw = self.bandwidth
        self.empty_data_prediction = None
        self.decay_rate = self.decay_rate if self.lookback_type != "cut-off" else -1
        self.is_exp = self.lookback_type == "exponential"
        self.inference_time = None

    def fit(self, reference) -> None:
        # Record the current time in seconds since the epoch
        fit_time_start = time.time()

        """Fit model with reference of a value x ordinal matrix."""
        self.ref = asarray(reference)
        self.var = self.ref.var(axis=0)
        self.empty_data_prediction = list(median(self.ref, axis=0))
        
        if self.lookback_type != "original":
            self.decay_table = self.create_decay_table()

        self.fit_time = timedelta(seconds=(time.time() - fit_time_start))
        # Simple performance metric based on fit time
        self._performance_metrics["fit time"] = self.fit_time.total_seconds()

    def predict_at(self, target: list, at):
        norm, weight = 0, 0
        if len(target):
            lookback = self.get_lookback_values(at)
        for m in range(self.ref.shape[0]):
            inner = 0
            for i in range(len(target)):
                # Handles missing TAN/TAF at inference time https://jira.dsb.dk/browse/TDP-1036
                if isnan(target[i]):
                    continue
                if (
                    self.lookback_type != "original"
                    and len(target)
                    and (lookback > i)
                    and (self.decay_table[-lookback + i] < 0.01)
                ):
                    continue
                product = (target[i] - self.ref[m, i]) ** 2 / self.var[i]
                if lookback > i and len(target):
                    inner += product * self.decay_table[-lookback + i]
                else:
                    inner += product
            kernel = e ** (-inner / self.bw)
            norm += kernel
            weight += kernel * (self.ref[m, at + 1] - self.ref[m, len(target)-1])  # delta delay from current stop
        
        if weight != weight or norm == 0 or norm != norm:
            if norm == 0:
                return target[-1]
            raise ValueError(f"{weight} / {norm}")
        
        return target[-1] + weight / norm

    def predict(self, context = None, target: array = array([])) -> List[float]:
        """
        Predict future values for target until reference length.
        Method must have 'context' input variable to be compatible with MLflow, even if unused.
        """
        target = target.tolist()

        if len(target) > 0 and isnan(target[-1]):
            raise ValueError("Train delay at current station cannot be NaN")
        if len(target) < 1:
            return self.empty_data_prediction
        else:
            forecast = (len(target) - 1) * [None] + [target[-1]]
        for i in range(len(target) - 1, self.ref.shape[1] - 1):
            forecast.append(self.predict_at(target, i))
            if self.recursive and len(target) > self.recursive_switch:
                target.append(forecast[-1])
        return forecast

    def set_inference_time(self, inference_time):
        self.inference_time = inference_time
        self._model_params["Inference time"] = inference_time

    def define_lookback(self, at):
        if at < self.lookback_cutoff:
            return 0
        else:
            return at - (self.lookback_cutoff - 1)

    def get_lookback_values(self, at):
        if (self.lookback_type == "original") or (self.lookback_type == "linear"):
            return 0
        return self.define_lookback(at)

    def create_decay_table(self):
        if self.decay_rate == -1:
            return [0 for t in range(self.ref.shape[1])]
        decay_table = (
            [
                exp_decay(t, reverse=True, halflife=self.decay_rate)
                for t in range(self.ref.shape[1])
            ]
            if self.is_exp
            else [
                lin_decay(t, n_zero=-1, reverse=True, steps=self.decay_rate)
                for t in range(self.ref.shape[1])
            ]
        )
        # TODO Avoid normalization
        return list(normalize([decay_table], norm="max").flatten())

    def collect_params(self):
        model_params = {}
        model_params["Super Route"] = str("HG-NÆ")
        model_params["Bandwidth"] = str(self.bw)
        model_params["Recursive"] = str(self.recursive)
        model_params["Recursive switch"] = str(self.recursive_switch)
        model_params["Lookback type"] = str(self.lookback_type)
        model_params["Lookback cutoff"] = str(self.lookback_cutoff)
        model_params["Decay rate"] = str(self.decay_rate)
        model_params["Inference time"] = str(self.inference_time)
        self._model_params = model_params
        return model_params

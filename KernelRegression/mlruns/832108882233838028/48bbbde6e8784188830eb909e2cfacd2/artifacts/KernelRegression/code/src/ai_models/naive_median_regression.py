from dataclasses import dataclass
from typing import List
from datetime import timedelta
import time
from src.ai_models.abstract_model import AbstractModel

@dataclass
class NaiveMedianRegression(AbstractModel):
    """Predict that the train is current delay + median of training data seconds delayed for each station"""
    
    model_name: str = "AI model Naive Median Regression"
    delay: int = 1
    lookback_cutoff: int = 0
    train_size: int = 500
    test_size: int = 500
    buffer_size: int = 10
    
    def __post_init__(self):
        # Data
        self.set_metrics()
        self.inference_time = None
    
    def fit(self, reference):
        """Fit model with reference of a value x ordinal matrix."""
        fit_time_start = time.time()
        self.ref = reference
        self.median = reference.median(axis=0)
        self.fit_time = timedelta(seconds=(time.time() - fit_time_start))
        # Simple performance metric based on fit time
        self._performance_metrics["fit time"] = self.fit_time.total_seconds()

    def predict(self, target):
        """Predict future values for target until reference length."""
        if len(target) < 1:
            # return [0.0] * self.ref.shape[1]
            return self.median.tolist()
        forecast = (
            (len(target) - 1) * [None]
            + (
                [
                    target[-1] - self.median.iloc[len(target) - 1]
                ]  # delay at current - median for current
                + self.median.iloc[
                    -(self.ref.shape[1] - (len(target) - 1)) :
                ]  # add median for target stations
            ).tolist()
        )
        return forecast
    

    def collect_params(self):
        model_params = {}
        model_params["Super Route"] = str("HG-NÆ")
        model_params["Delay"] = str(self.delay)
        self._model_params = model_params
        return model_params
    
    def set_inference_time(self, inference_time):
        self.inference_time = inference_time
        self._model_params["Inference time"] = inference_time
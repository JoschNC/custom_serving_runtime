from src.ai_models.abstract_model import AbstractModel, ModelCategory
from collections import OrderedDict
import time
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class NullModel(AbstractModel):
    """Predict that the train is current delay + median of training data seconds delayed for each station"""
    name: str
    superroute: str
    category: ModelCategory = ModelCategory.SUPERROUTE
    train_size: int = 500  # 180
    test_size: int = 500
    buffer_size: int = 10
    artifacts: OrderedDict = field(default_factory=OrderedDict)
    tables: OrderedDict = field(default_factory=OrderedDict)

    def __post_init__(self):
        super().__init__(name = self.name, category = self.category, superroute = self.superroute)
        self.set_metrics()

    def fit(self, reference):
        """Fit model with reference of a value x ordinal matrix."""
        fit_time_start = time.time()
        self.ref = reference
        self.fit_time = timedelta(seconds=(time.time() - fit_time_start))
        # Simple performance metric based on fit time
        self._performance_metrics["fit time"] = self.fit_time.total_seconds()

    def predict(self, context = None, target = []):
        """
        Predict future values for target until reference length.
        Method must have 'context' input variable to be compatible with MLflow, even if unused.
        """
        if len(target) < 1:
            return [0.0] * self.ref.shape[1]
        forecast = (len(target) - 1) * [None] + [target[-1]] * (
            self.ref.shape[1] - (len(target) - 1)
        )
        return forecast

    def set_inference_time(self, inference_time):
        self.inference_time = inference_time

    def collect_params(self):
        model_params = {}
        model_params["Super Route"] = str("HG-NÆ")
        model_params["Inference time"] = str(self.inference_time)
        self._model_params = model_params
        return model_params

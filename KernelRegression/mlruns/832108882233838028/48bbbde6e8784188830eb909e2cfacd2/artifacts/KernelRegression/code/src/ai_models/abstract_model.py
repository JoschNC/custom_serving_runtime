# Imports
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import ModelSignature
from pandas import DataFrame, Series
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Union
from enum import Enum

class ModelCategory(Enum):
    SUPERROUTE = "superroute"

class AbstractModel(PythonModel, ABC):
    """An abstract model for other models to inherit from."""

    def __init__(self, name: str, category: Union[str, ModelCategory], superroute: Optional[str] = None) -> None:
        """
        Initialize the AbstractModel.

        Args:
            name (str): The name of the model.
            category (str): The category (SuperRoute, Route) of the model.

        Attributes:
            name (str): The name of the model.
            signature (Optional[ModelSignature]): The model signature, initially set to None.
            category (str): The category (SuperRoute, Route) of the model.
            superroute (Optional[str]): The superroute (the identifier string, e.g. "HG-NÆ") of the model, must be provided if the category is SuperRoute.
        """
        self.name = name
        self.signature: Optional[ModelSignature] = None
        self.category: ModelCategory = self._parse_category(category)
        self.superroute: Optional[str] = superroute

        if self.category == ModelCategory.SUPERROUTE:
            if self.superroute is None:
                raise ValueError("Superroute is required for SuperRoute models.")

        
        self.set_metrics()


    @abstractmethod  # UPDATED
    def fit(self, X: DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, context, X: DataFrame) -> Series:
        pass

    @abstractmethod
    def collect_params(self) -> dict:
        pass

    @staticmethod
    def _parse_category(category: Union[str, ModelCategory]) -> ModelCategory:
        if isinstance(category, ModelCategory):
            return category
        try:
            return ModelCategory(category.lower())
        except ValueError:
            raise ValueError(f"Invalid category: {category}. Must be one of {[c.value for c in ModelCategory]}")
    
    def get_metadata(self) -> dict:
        """
        Get metadata about the model, including superroute if applicable.
        """
        metadata = {
        }
        if self.category == ModelCategory.SUPERROUTE and self.superroute:
            metadata["superroute"] = self.superroute
        return metadata

    def set_metrics(self):
        # Metrics
        self.fit_time = None
        self.inference_time = None
        self._performance_metrics = OrderedDict()
        self._model_params = OrderedDict()  # about model
        self._metadata = OrderedDict()  # about the training data
        self._artifacts = OrderedDict()
        self._tables = OrderedDict()

    # TODO merge metrics save methods into one, avoid repeating code.
    def save_performance_metrics_old(self, old_table, status_code_string = "All"):
        mae = old_table["MAE"]
        rmse = old_table["RMSE"]
        for ind in range(len(mae)):
            if str(mae.index[ind]) == "NCKR 9+ MIN":
                self._performance_metrics[f"old_metric_MAE_9PLUSmin_{status_code_string}"] = mae.iloc[ind]
            else:
                self._performance_metrics[
                    "old_metric_MAE_" + str(mae.index[ind]) + "_" + status_code_string
                ] = mae.iloc[ind]
        for ind in range(len(rmse)):
            if str(rmse.index[ind]) == "NCKR 9+ MIN":
                self._performance_metrics[f"old_metric_RMSE_9PLUSmin_{status_code_string}"] = rmse.iloc[ind]
            else:
                self._performance_metrics[
                    "old_metric_RMSE_" + str(rmse.index[ind]) + "_" + status_code_string
                ] = rmse.iloc[ind]

    # TODO merge metrics save methods into one, avoid repeating code.
    def save_performance_metrics_new(self, new_table, status_code_string = "All"):
        row_names = [
            "All measurements",
            "Already Delayed",
            "Minor Delays",
            "On Plan",
            "Sudden Disruption",
        ]
        types_metric = ["MAE", "RMSE"]
        stops_ahead = ["All", "1-3", "4-6", "7-9"]
        nmt = new_table

        for name in row_names:
            for metric in types_metric:
                for stop in stops_ahead:
                    self._performance_metrics[
                        name + "_" + metric + "_" + stop + "_" + status_code_string
                    ] = nmt[nmt.index == name][stop][metric].values[0]

    @property
    def artifacts(self):
        return self._artifacts

    @artifacts.setter
    def artifacts(self, artifacts: dict):
        self._artifacts = artifacts

    @property
    def tables(self):
        return self._tables

    @tables.setter
    def tables(self, tables: dict):
        self._tables = tables

    @property
    def performance_metrics(self) -> dict:
        return self._performance_metrics

    @performance_metrics.setter
    def performance_metrics(self, metrics: dict):
        self._performance_metrics = metrics

    @property
    def model_params(self) -> dict:
        return self._model_params

    @model_params.setter
    def model_params(self, params: dict):
        self._model_params = params

    def collect_metadata(self, train_data, test_data) -> dict:
        model_metadata = {}
        model_metadata["Training data start date"] = str(
            train_data.index.get_level_values(0).min()
        )
        model_metadata["Training data end date"] = str(
            train_data.index.get_level_values(0).max()
        )
        model_metadata["Training data size"] = str(self.train_size)
        model_metadata["Test data start date"] = str(
            test_data.index.get_level_values(0).min()
        )
        model_metadata["Test data end date"] = str(
            test_data.index.get_level_values(0).max()
        )
        model_metadata["Test data size"] = str(self.test_size)
        self._metadata = model_metadata
        return model_metadata

    def prepare_data(self, unprepared_data: DataFrame) -> DataFrame:
        """
        Prepare the input data for model training or prediction.

        This function performs the following steps:
        1. Creates a 'stop' column by combining measurementIndex, trackSectionStation, and statusCode.
        2. Pivots the data to create a wide format with 'stop' as columns and 'actualMinusPlanned' as values.
        3. Sets the index to 'first_departure', 'trainCategory', and 'trainNumber'.
        4. Reorders columns based on the original 'stop' order.

        Args:
            unprepared_data (DataFrame): The raw input data containing train delay information.

        Returns:
            DataFrame: A prepared DataFrame with stops as columns and delay values as entries,
                    indexed by departure time, train category, and train number.
        """
        unprepared_data["stop"] = (
            unprepared_data["measurementIndex"].astype(str)
            + " "
            + unprepared_data["trackSectionStation"]
            + " ("
            + unprepared_data["statusCode"]
            + ")"
        )
        column_order = unprepared_data["stop"].drop_duplicates()
        return (
            unprepared_data[
                [
                    "stop",
                    "actualMinusPlanned",
                    "trainNumber",
                    "trainCategory",
                    "first_departure",
                ]
            ]
            .pivot(
                columns="stop",
                values="actualMinusPlanned",
                index=[
                    "first_departure",
                    "trainCategory",
                    "trainNumber",
                ],
            )
            .reindex(column_order, axis=1)
        )

    def train_test_split(self, data: DataFrame) -> tuple[DataFrame, DataFrame]:
        if len(data) < self.train_size + self.buffer_size + self.test_size:
            raise ValueError(
                f"""
                Not enough data, found length of data: [{len(data)}], less than sum of
                train_size: {self.train_size}, buffer_size: {self.buffer_size}, and test_size: {self.test_size}
                """
            )
        train_data = data[: self.train_size]
        test_data = data[
            self.train_size + self.buffer_size : self.train_size
            + self.buffer_size
            + self.test_size
        ]
        if len(train_data) != self.train_size:
            raise ValueError(
                f"expected [len(train_data) == train_size], got [{len(train_data)} != {self.train_size}]"
            )
        if len(test_data) != self.test_size:
            raise ValueError(
                f"expected [len(test_data) == test_size], got [{len(test_data)} != {self.test_size}]"
            )
        return train_data, test_data

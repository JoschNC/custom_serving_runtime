from src.ai_models.abstract_model import AbstractModel, ModelCategory
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from azure.eventhub import EventHubProducerClient
from azure.eventhub import EventData
from azure.eventhub.exceptions import AuthenticationError, EventHubError
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventTypeSchema(Enum):
    MODEL_ADDED = "model_added"
    MODEL_DELETED = "model_deleted"

class MessageModelInfoSchema(BaseModel):
    name: str = Field(..., description="The name of the model")
    version: str = Field(..., description="The version of the model")
    uri: str = Field(..., description="Path to the models .pkl file in Data Lake Storage")
    model_category: ModelCategory = Field(..., description="The category (e.g. Super route, Route, etc.) of the model")

class EventMessageSchema(BaseModel):
    event_type: EventTypeSchema = Field(..., description="The type of event")
    timestamp: datetime = Field(..., description="The timestamp of the event")
    model_info: MessageModelInfoSchema = Field(..., description="The information about the model")
    metadata: Optional[dict] = Field(None, description="Additional metadata for the model")


class ModelPublisher:
    def __init__(
        self,
        model: AbstractModel,
        model_version: ModelVersion,
        mlflow_client: MlflowClient,
        event_hub_producer: Optional[EventHubProducerClient] = None,
    ):
        """
        Initializes the ModelPublisher with a model, an MLflow client, and an optional EventHubProducerClient.
         Example usage:
        ```python
        from mlflow import set_tracking_uri, set_registry_uri, set_experiment
        from mlflow.models.signature import ModelSignature
        from mlflow.tracking import MlflowClient
        from src.ai_models.registry.ModelPublisher import ModelPublisher
        from src.ai_models.kernel_regression import KernelRegression

        # Instantiate the model 
        model = KernelRegression(name="example_model", category="superroute", superroute="KBH-AAB")

        # Set up MLflow
        set_tracking_uri("databricks")
        set_registry_uri('databricks-uc')
        set_experiment("/example_experiment")

        # Create MLflow client
        mlflow_client = MlflowClient()

        # Set model signature
        model.signature = ModelSignature.from_dict({
            'inputs': '[{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1]}}]',
            'outputs': '[{"type": "double", "required": true}]',
            'params': None
        })

        # Create ModelPublisher instance
        publisher = ModelPublisher(model=model, mlflow_client=mlflow_client)

        # Publish the model
        success = publisher.publish()

        print(f"Model published successfully: {success}")
        ```
        Args:
            model (AbstractModel): The model to be published.
            model_version (ModelVersion): The MLflow ModelVersion.
            mlflow_client (MlflowClient): The MLflow tracking client.
            event_hub_producer (Optional[EventHubProducerClient], optional): Pre-configured EventHubProducerClient. Defaults to None.
        """
        self.model = model
        self.model_version = model_version
        self.mlflow_client = mlflow_client
        self.event_hub_producer = event_hub_producer or self._configure_event_hub()

    def publish(self) -> bool:
        """
        Sends an event message to the EventHub that a new Champion exists.

        Returns:
            bool: True if publish is successful, False otherwise.
        """
        try:
            artifacts_path = self._get_artifacts_path()
            message = self._create_model_added_message(artifacts_path)
            self._send_message(message)
            return True
        except Exception as e:
            logger.error(f"An error occurred during the publish operation: {e}", exc_info=True)
            return False

    def _get_artifacts_path(self):
        artifacts_path = self.mlflow_client.get_model_version_download_uri(
            self.model_version.name, self.model_version.version
        )
        logger.debug(f"Artifacts path: {artifacts_path}")
        return artifacts_path

    def _create_model_added_message(self, artifacts_path):
        return self._construct_event_message(
            event_type=EventTypeSchema.MODEL_ADDED,
            model_name=self.model_version.name,
            model_version=self.model_version.version,
            model_category=self.model.category,
            uri=artifacts_path,
            metadata={}
        )

    def _configure_event_hub(
        self,
        connection_string: Optional[str] = None,
        event_hub_name: Optional[str] = None
    ) -> EventHubProducerClient:
        """
        Configures the Event Hub producer client.

        Args:
            connection_string (Optional[str], optional): Event Hub connection string. Defaults to None.
            event_hub_name (Optional[str], optional): Event Hub name. Defaults to None.

        Returns:
            EventHubProducerClient: Configured EventHubProducerClient instance.
        """
        if not 'dbutils' in globals():
        # We're in a Databricks Connect Context
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            dbutils = w.dbutils
        
        #TO DO: This should be moved to a secret manager that we can use across environments
        conn_str = connection_string or dbutils.secrets.get(scope="kv-dlh-aiml-ttidelay-dev", key="conn-str-ehpulse-dsb-tti-ns")
        hub_name = event_hub_name or dbutils.secrets.get(scope="kv-dlh-aiml-ttidelay-dev", key="name-ehpulse-dsb-tti-ns")

        if not conn_str or not hub_name:
            raise ValueError("Event Hub connection string and name must be provided either as arguments or secrets.")

        return EventHubProducerClient.from_connection_string(
            conn_str=conn_str,
            eventhub_name=hub_name
        )

    def _construct_event_message(
        self,
        event_type: EventTypeSchema,
        model_name: str,
        model_version: str,
        uri: str,
        model_category: ModelCategory,
        metadata: Optional[dict] = None
    ) -> EventMessageSchema:
        """
        Constructs an EventMessage object with the provided details.

        Args:
            event_type (EventType): The type of event.
            model_name (str): The name of the model.
            model_version (str): The version of the model.
            uri (str): URI where the model is stored.
            metadata (Optional[dict], optional): Additional metadata. Defaults to None.

        Returns:
            EventMessage: The constructed event message.
        """
        if metadata is None:
            metadata = {}
        model_metadata = self.model.get_metadata()

        metadata.update(model_metadata)
    
        return EventMessageSchema(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            model_info=MessageModelInfoSchema(name=model_name, version=model_version, uri=uri, model_category=model_category),
            metadata=metadata
        )
    
    def _send_message(self, message: EventMessageSchema) -> None:
        """
        Sends an event message to the configured Event Hub.

        Args:
            message (EventMessage): The event message to send.
        """
        logger.debug(f"Message to be sent: {message.model_dump_json()}")

        try:
            with self.event_hub_producer as producer:
                producer.send_event(EventData(message.model_dump_json()))
                logger.info("Message sent to Event Hub successfully.")

        except AuthenticationError as auth_error:
            logger.error(f"Authentication failed when sending message to Event Hub: {auth_error}", exc_info=True)
        except EventHubError as hub_error:
            logger.error(f"Event Hub error: {hub_error}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred when sending message to Event Hub: {e}", exc_info=True)
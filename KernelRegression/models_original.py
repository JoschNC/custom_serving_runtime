import mlserver
from mlserver import MLModel
from mlserver.utils import get_model_uri
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
import mlflow
import numpy as np
import pandas as pd
import requests
import numpy as np
import json

class CustomMLflowModel(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        self.model = mlflow.pyfunc.load_model(model_uri)
        self._ready = True
        return self._ready


    async def predict(self, payload: InferenceRequest) -> InferenceResponse:

        print("Payload:")
        print(payload)
        print()
        # Unpack InferenceRequest
        payload_input = payload.inputs[0]

        print("Payload_input")
        print(payload_input)
        print()

        # Unpack RequestInput
        data = payload_input.data

        print_variable(data, "data")

        # Unpack array
        array = data.root
   
        np_arr = np.array(array, dtype=np.int32)

        print(np_arr)

        model_output = self.model.predict(np_arr)

        # Changing None values to NaN
        model_output = np.array([np.nan if x is None else x for x in model_output], dtype= np.float64)

        print_variable(model_output, "model_output")
        
        response_output = ResponseOutput(name="response",
                                         shape = [1,len(model_output)],
                                         datatype="FP64",
                                         data = model_output)
        print("Response output:")
        print(response_output)
        print()
        inference_response = InferenceResponse(model_name="test", outputs = [response_output])

        return inference_response


class CustomMLflowModelNoPrint(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        self.model = mlflow.pyfunc.load_model(model_uri)
        self._ready = True
        return self._ready


    async def predict(self, payload: InferenceRequest) -> InferenceResponse:

        # Unpack InferenceRequest
        payload_input = payload.inputs[0]

        # Unpack RequestInput
        data = payload_input.data

        # Unpack array
        array = data.root
   
        np_arr = np.array(array, dtype=np.int32)

        model_output = self.model.predict(np_arr)

        json_response_data = json.dumps({"response_array":model_output}).encode("utf-8")
        
        response_output = ResponseOutput(name="response",
                                         shape = [len(json_response_data)],
                                         datatype="BYTES",
                                         data = [json_response_data])

        inference_response = InferenceResponse(model_name="test", outputs = [response_output])

        return inference_response

def print_variable(variable, name):
    print(f"{name}:")
    print(variable)
    print()
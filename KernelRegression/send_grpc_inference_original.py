import requests
import json
import grpc
from mlserver.codecs.string import StringRequestCodec
import mlserver.grpc.converters as converters
import mlserver.grpc.dataplane_pb2_grpc as dataplane
import mlserver.types as types
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=1)

model_name = "KernelRegression"
input_array = [1, 2, 3, 4]

inference_request = types.InferenceRequest(
    inputs=[
        types.RequestInput(
            name="a_name",
            shape=[len(input_array)],
            datatype="FP32",
            data=input_array
        )
    ]
)

inference_request_g = converters.ModelInferRequestConverter.from_types(
    inference_request, model_name=model_name, model_version=None
)
grpc_channel = grpc.insecure_channel("0.0.0.0:8081")
grpc_stub = dataplane.GRPCInferenceServiceStub(grpc_channel)

response = grpc_stub.ModelInfer(inference_request_g)

print(f"full response:\n")
print(response)
print()
# retrive text output as dictionary
inference_response = converters.ModelInferResponseConverter.to_types(response)
raw_json = StringRequestCodec.decode_response(inference_response)
output = json.loads(raw_json[0])
print(f"\ndata part:\n")
pp.pprint(output)
import pandas as pd
import json
import numpy as np
import http.client
from urllib.parse import urlparse
from mlserver.types import InferenceRequest, RequestInput
from mlserver.codecs import NumpyCodec

arr = [1, 2, 3, 4]

# Convert array to JSON
arr_json = json.dumps({"arr": arr})

# Create the RequestInput
request_input = RequestInput(name="request",
                             shape=[-1],
                             datatype="INT32",
                             data=arr_json)
print("Inference input:")
print(request_input)
print()

# Encode the RequestInput into an InferenceRequest
inference_request = InferenceRequest(inputs=[request_input])

print("Inference request:")
print(inference_request)
print()

# Serialize the InferenceRequest to JSON
inference_request_json = inference_request.model_dump(by_alias=True)

# Pretty-print the JSON body of the request
request_body = json.dumps(inference_request_json, indent=2)
print("Serialized Inference Request JSON body:")
print(request_body)
print()

# Define the endpoint
endpoint = "http://localhost:8080/v2/models/KernelRegression/infer"  # CHANGE TO RELEVANT INPUT

# Parse the URL
url = urlparse(endpoint)
conn = http.client.HTTPConnection(url.hostname, url.port)

# Define headers
headers = {
    'Content-Type': 'application/json'
}

# Print the complete HTTP request
print("Complete HTTP request:")
print(f"POST {url.path} HTTP/1.1")
print(f"Host: {url.hostname}:{url.port}")
for header, value in headers.items():
    print(f"{header}: {value}")
print()
print(request_body)
print()

# Send the request
conn.request("POST", url.path, body=request_body, headers=headers)
response = conn.getresponse()

# Read and print the response
response_body = response.read().decode()
print("Inference response:")
print(response_body)
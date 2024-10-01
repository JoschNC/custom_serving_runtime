import mlflow
import numpy as np

arr = np.array([1,2,3,4], dtype=np.int32)

model = mlflow.pyfunc.load_model('mlruns/2/a9dbbbe122d74fddb495e72d7077b3eb/artifacts/KernelRegression')

result = model.predict(arr)

print(result)
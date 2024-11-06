from fastapi import FastAPI
import uvicorn
import numpy as np
import mlflow
from pydantic import BaseModel
import utils_mlflow


class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class NewModel(BaseModel):
    model_name: str
    model_version: str


def load_model_ws(base_uri, model_name, model_version):
    mlflow.set_tracking_uri(uri=base_uri)

    model_uri = utils_mlflow.get_model_uri(model_name, model_version)
    return mlflow.sklearn.load_model(model_uri)


app = FastAPI()
base_uri = utils_mlflow.get_base_uri()
mlflow.set_tracking_uri(uri=base_uri)
model = load_model_ws(base_uri, "tp_mlflow_mlops", "latest")


@app.post("/predict")
async def read_root(item: Iris):
    sepal_length = int(item.sepal_length)
    sepal_width = int(item.sepal_width)
    petal_length = int(item.petal_length)
    petal_width = int(item.petal_width)

    predictions = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))

    return {"prediction": predictions.tolist()}


@app.post("/update-model")
async def update_model(item: NewModel):
    global model
    try:
        model = load_model_ws(item.model_name, item.model_version)
        return {"status": "Model updated"}
    except Exception as e:
        return {"status": "Model update failed", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1234)

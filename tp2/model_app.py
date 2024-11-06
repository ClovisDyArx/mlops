from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import joblib
import numpy as np

def load_model():
    return joblib.load("regression.joblib")

class Item(BaseModel):
    size: float
    nb_rooms: int
    garden: bool

app = FastAPI()
model = load_model()

@app.post("/predict")
async def read_root(item: Item):
    garden_int = int(item.garden)
    features = np.array([[item.size, item.nb_rooms, garden_int]])
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6969)

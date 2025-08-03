from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

model = torch.load("model.pth")

fast_app = FastAPI()

class InputData(BaseModel):
    features: list[float]  # or list[list[float]] if batch

@fast_app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
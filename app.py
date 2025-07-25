from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

model = xgb.XGBRegressor()
model.load_model("final_model.xgb")

fast_app = FastAPI()

class InputData(BaseModel):
    features: list[float]  # or list[list[float]] if batch

@fast_app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
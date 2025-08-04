from fastapi import FastAPI
from pydantic import BaseModel
from rnn import RNN
from deep_learning_data import community_to_id, scaler
import torch
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 30
input_size = 4
hidden_size = 64
num_layers = 4
learning_rate = .001
 
model = RNN(input_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load("model.pth", weights_only = True, map_location = device))
model.eval()

app = FastAPI()

class InputData(BaseModel):
    graffiti: int
    potholes: int
    year: int
    month: int
    community_id: str

@app.post("/predict")
def predict(data: InputData):
    community = community_to_id[data.community_id]
    community_tensor = torch.tensor([community], dtype = torch.long).to(device)

    input_array = np.array([[[data.graffiti, data.potholes, data.year, data.month]]], dtype = np.float32)
    input_tensor = torch.tensor(input_array)
    input_tensor = input_tensor.to(device)

    hidden = model.init_hidden(batch_size = 1)

    with torch.no_grad():
        output = model(input_tensor, hidden, community_tensor)

    output = output.cpu().detach().numpy()
    output = float(scaler.inverse_transform(output))
    return {"prediction": output}
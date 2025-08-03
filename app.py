from fastapi import FastAPI
from pydantic import BaseModel
from rnn import RNN
from deep_learning_data import community_tensor
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 30
input_size = 4
hidden_size = 64
num_layers = 4
learning_rate = .001

model = RNN(input_size, hidden_size, num_layers).to(device)
model = torch.load_state_dict(torch.load("model.pth", map_location = device))
model.eval()

app = FastAPI()

class InputData(BaseModel):
    features: list[list[float]]
    community_id: str

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.features, dtype=np.float32).reshape(1, -1)
    input_tensor = torch.tensor(input_array)

    input_tensor = input_tensor.to(device)
    model = model.to(device)
    community_ids = torch.tensor([data.community_id], dtype=torch.long).to(device)

    hidden = model.init_hidden(batch_size = 1)

    with torch.no_grad():
        output = model(input_tensor, hidden, community_ids)

    prediction = output.numpy().tolist()
    return {"prediction": prediction}
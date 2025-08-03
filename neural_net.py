import torch
from torch import nn
from torch import functional as f
import torch.optim as optim
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost_testing import df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# creating the neural network class
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 18),
            nn.ReLU(),
            nn.Linear(18, 1)) 

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


df["community"] = df["community"].astype("category").cat.codes

x = df.drop(columns=["previous_month", "realized_crime", "true_month"])
y = df["realized_crime"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)   
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)


input_size = x_train_tensor.shape[1]
model = NeuralNet(input_size).to(device)

model(x_train_tensor)

loss_fn  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch = 200

train_losses = []

for e in range(0, epoch):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    if e % 2 == 0:
        print(f"Epoch {e}, Loss: {loss.item()}")

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve")
plt.show()

#I think this is the wrong model because it is reading the whole dataset at once, instead of 
# differentiating between the time series nature of the months. Lets try LSTM or RNN.
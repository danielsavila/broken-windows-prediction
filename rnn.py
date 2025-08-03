import os
import torch
from torch import functional as f
from torchinfo import summary
from torch import nn, optim
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deep_learning_data import x_train, x_test, y_train, y_test, community_tensor

#setting up environmental variables for rnn
#torch.use_deterministic_algorithms(True) #for reproducability
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #for known np package error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 123456


#instantiating RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #use an embedding for the communities column, since categorical
        self.embedding = nn.Embedding(77, 4)

        self.rnn = nn.RNN(input_size + 4, hidden_size, num_layers, batch_first = True)
        #self.fc is the last layer, and returns a regression prediction, so final output is 1 "class"
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden, community_ids):
        emb = self.embedding(community_ids)
        emb_expanded = emb.unsqueeze(1)
        emb_expanded = emb_expanded.repeat(1, x.size(1), 1)
        concat_tensor = torch.cat((x, emb_expanded), dim = 2)
        out, hidden = self.rnn(concat_tensor, hidden)
        output = self.fc(out[:, -1, :])
        return output
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)
    



#training the model
batch_size = int(round(x_train.shape[0]/10, 0))
input_size = 4
hidden_size = 64
num_layers = 4
learning_rate = .01
    
model = RNN(input_size, hidden_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
summary(model)

torch.manual_seed(seed)

it_history = []
num_epochs = 1000
for epoch in range(num_epochs):
    hidden = model.init_hidden(batch_size)
    epoch_loss = 0.0

    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i+batch_size].to(device).unsqueeze(1)
        batch_community_tensor = community_tensor[i:i + batch_size].to(device)
        hidden = model.init_hidden(batch_x.size(0))

        outputs = model(batch_x, hidden, batch_community_tensor)

        batch_y = y_train[i:i + batch_size].to(device)
        batch_loss = criterion(outputs, batch_y.reshape(-1))
        epoch_loss += batch_loss.item()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    it_history.append([epoch, epoch_loss])
    print(epoch, round(epoch_loss, 2))

df_history = pd.DataFrame(it_history, columns = ['Epoch', 'Epoch Loss'])

plt.figure(figsize = (12,6), dpi = 200)
plt.plot(df_history['Epoch'], df_history['Epoch Loss'], color = 'royalblue')
plt.xlabel('Epoch')
plt.ylabel('Epoch Loss')
plt.grid(axis = 'y')
plt.show()
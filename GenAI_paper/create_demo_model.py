# create_demo_model.py
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)
    
    def forward(self, state):
        x = state
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

# Create a demo model
model = DQN(13, 10)
torch.save(model.state_dict(), 'final_model.pth')
print("Demo model created!")
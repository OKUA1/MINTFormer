from torch import nn
from torch.nn import functional as F

class Autoencoder(nn.Module):
    def __init__(self, inp_size):
        super(Autoencoder, self).__init__()
        hidden_size = inp_size // 3
        self.fc_1 = nn.Linear(inp_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, inp_size)

    def forward(self, x, params=None):
        if params is None:
            # Use the model's parameters
            weight1, bias1 = self.fc_1.parameters()
            weight2, bias2 = self.fc_2.parameters()
        else:
            # Use the provided parameters
            #print(params, len(params))
            weight1, bias1, weight2, bias2 = params

        out = F.relu(x @ weight1.t() + bias1)
        out = out @ weight2.t() + bias2
        return out
    

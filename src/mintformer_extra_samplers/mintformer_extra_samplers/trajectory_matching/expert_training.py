import torch 
from torch import nn
import numpy as np
from .model import Autoencoder
from tqdm import tqdm
import uuid
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.MSELoss().to(DEVICE)

def train_experts(X: np.ndarray, n_experts: int = 3, batch_size: int = 128, n_epochs:int = 100):
    input_size = X.shape[1]
    uuid_val = str(uuid.uuid4())
    print(uuid_val, type(uuid_val))
    
    for i in range(n_experts):
        expert_path = os.path.join("experts", uuid_val, str(i))
        os.makedirs(expert_path, exist_ok=True)
        model = Autoencoder(input_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        with tqdm(total=n_epochs, desc=f"Expert {i+1}") as pbar:
            for epoch in range(n_epochs):
                running_loss = 0.0
                total_samples = 0
                for batch in dataloader:
                    optimizer.zero_grad()

                    inputs, y = batch
                    reconstructed = model(inputs)
                    loss = criterion(y, reconstructed)

                    loss.backward()
                    optimizer.step()
                    total_samples += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                average_loss = running_loss / total_samples
                pbar.set_postfix(loss=average_loss)
                pbar.update()
                torch.save(model.state_dict(), os.path.join(expert_path, f"{epoch}"))
    
    return os.path.join("experts", uuid_val)
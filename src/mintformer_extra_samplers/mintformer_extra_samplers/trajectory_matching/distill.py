import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
import copy
import random
from .model import Autoencoder


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def flatten_parameters(model):
    params = list(model.parameters())
    shapes = [param.shape for param in params]
    flat_params = torch.cat([param.view(-1) for param in params]).requires_grad_(True).to(DEVICE)
    return flat_params, shapes

def unflatten_parameters(flat_params, shapes):
    params = []
    i = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape)).item()
        param = flat_params[i:i+size].view(shape)
        params.append(param)
        i += size
    return params

def distill(n_samples, n_features, syn_steps, expert_epochs, n_experts, expert_path, batch_size, range_multiplier: int = 2, n_iter = 10):
    
    N = syn_steps
    M = syn_steps * range_multiplier
    starting_pos_upper_bound = expert_epochs - M

    data_syn = torch.rand(size = (n_samples, n_features), dtype=torch.float)

    syn_lr = torch.tensor(0.001).to(DEVICE)

    ##### training

    data_syn = data_syn.detach().to(DEVICE).requires_grad_(True)
    syn_lr = syn_lr.detach().to(DEVICE).requires_grad_(True)

    optimizer = torch.optim.Adam([data_syn], lr = 0.001)
    optimizer_lr = torch.optim.Adam([syn_lr], lr = 0.001)

    criterion = nn.MSELoss().to(DEVICE)
    print(data_syn)
    for i in tqdm(range(n_iter)):
        # print(f"Iteration: {i+1}")

        # pick a random expert and t

        start_e = np.random.randint(0, n_experts)
        start_t = np.random.randint(0, starting_pos_upper_bound)

        tN, tM = start_t + N, start_t + M

        student = Autoencoder(n_features).to(DEVICE)
        teacher_t = Autoencoder(n_features).to(DEVICE)
        teacher_tM = Autoencoder(n_features).to(DEVICE)

        student.load_state_dict(torch.load(os.path.join(expert_path, str(start_e), str(start_t))))
        teacher_t.load_state_dict(torch.load(os.path.join(expert_path, str(start_e), str(start_t))))
        teacher_tM.load_state_dict(torch.load(os.path.join(expert_path, str(start_e), str(tM))))
        params, shapes = flatten_parameters(student)
        student_params = [params]
        starting_params, _ = flatten_parameters(teacher_t)
        target_params, _ = flatten_parameters(teacher_tM)
        num_params = sum([np.prod(p.size()) for p in (student.parameters())]) 
        for step in range(N):
            indices = torch.randperm(n_samples).to(DEVICE)
            indices_chunks = list(torch.split(indices, batch_size))
            selected_indices = indices_chunks.pop(0)

            x = data_syn[selected_indices]
            y = data_syn[selected_indices].detach().to(DEVICE)

            yh = student(x, params = unflatten_parameters(student_params[-1], shapes))
            loss = criterion(y, yh)

            grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(DEVICE)
        param_dist = torch.tensor(0.0).to(DEVICE)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer.step()
        optimizer_lr.step()

    npt = data_syn.cpu().detach().numpy()
    
    return npt
import torch.nn as nn


def get_base_model():
    layers = []
    layers.append(nn.Linear(62, 42))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(42, 16))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(16, 2))
    layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)
    return model
 

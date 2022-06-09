import torch

def build_Adam(params,lr,weight_decay):
    return torch.optim.Adam(params=params,lr=lr,weight_decay=weight_decay)

def build_SGD(params,lr,weight_decay):
    return torch.optim.SGD(params=params,lr=lr,momentum=0.1,weight_decay=weight_decay)
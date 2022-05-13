import torch
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
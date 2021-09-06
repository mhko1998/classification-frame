import torch

def getoptim(net):
    optimizer1 = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9,0.999), eps=1e-08, weight_decay=0,amsgrad=True)
    return optimizer1
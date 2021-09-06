import torch
import getnet

net1 = getnet.Net()
optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.0001, betas=(0.9,0.999), eps=1e-08, weight_decay=0,amsgrad=True)
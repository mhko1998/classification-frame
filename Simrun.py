import torch
import dataloader
import getnet
import getoptim
import torch.nn as nn
import training1
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
def Simrun():
    device=torch.device('cuda:2' if torch.cuda.is_available() else'cpu')
    trainloader,testloader=dataloader.data_loader()
    net1 = getnet.Net()
    net1 = net1.to(device)


    criterion = nn.CrossEntropyLoss()

    optimizer=getoptim.getoptim(net1)

    for epoch in range(21):
        training1.train(net1,trainloader,optimizer,criterion,device,epoch)

        if epoch % 5 == 0:
            training1.test(net1,testloader,device)
    print('Finished Training')
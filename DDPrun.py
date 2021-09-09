import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import getnet
import getoptim
import training1
import dataloader
import os
from torch.nn.parallel import DistributedDataParallel as DDP

def DDPrun(gpu, args):
    rank=args.nr*args.gpus+gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    
    torch.cuda.set_device(gpu)
    trainloader,testloader=dataloader.DDP_data_loader(rank,args.world_size)
    
    net=getnet.Net().cuda(gpu)
    ddp_model=DDP(net,device_ids=[gpu])

    criterion=nn.CrossEntropyLoss().cuda(gpu)
    optimizer=getoptim.getoptim(ddp_model)
        
    for epoch in range(21):
        training1.DDPtraining(rank,ddp_model,trainloader,optimizer,criterion,epoch)
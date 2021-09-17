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
import loading
import os
import neptune.new as neptune
from torch.nn.parallel import DistributedDataParallel as DDP

def DDPrun(rank, args):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    
    trainloader, testloader = dataloader.DDP_data_loader(rank, args.world_size)
    
    net=getnet.Net().cuda(rank)
    ddp_model=DDP(net,device_ids=[rank])

    criterion=nn.CrossEntropyLoss().cuda(rank)
    optimizer=getoptim.getoptim(ddp_model)
    if args.load==True:
        loading.LOAD(ddp_model,optimizer)
    run=neptune.init(api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOGQ5Y2U4OC0xZWIzLTQyZjQtYWIyMy0wNTA5N2ExMzg2N2IifQ==',project='mhko1998/class')
    max=0
    for epoch in range(args.num_epochs):
        a=training1.DDPtraining(rank, ddp_model, trainloader, optimizer, criterion,epoch)
        run["loss"].log(a)
        if epoch%5==0:
            b=training1.DDPtest(rank,ddp_model,testloader,optimizer)
            run["acc"].log(b)
        if max<b:
            max=b
            loading.SAVE(ddp_model,optimizer,epoch)
            
    dist.destroy_process_group()

import torch
import torch.nn as nn
from torch.optim import optimizer
import getnet
import getoptim
import main
import os
from copy import deepcopy

def SAVE(net,optim,epoch,args):
    if args.world_size>1:
        model_param = deepcopy(net.module.state_dict())
    else:
        model_param = deepcopy(net.state_dict())
    param_dict={'model':model_param, 'optimizer':optim.state_dict(), 'save_epoch':epoch}
    save_file_name=os.path.join(args.pathdir, 'save.pt')
    torch.save(param_dict,save_file_name)

def LOAD(net,optim,args):
    module = torch.load(os.path.join(args.pathdir,'save.pt'))
    net.load_state_dict(module['model'])
    optim.load_state_dict(module['optimizer'])
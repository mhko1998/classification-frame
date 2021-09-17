import torch
import torch.nn as nn
import getnet
import getoptim
import main

def SAVE(net,optim,epoch):
    param_dict={'model':net.state_dict(),'optimizer':optim.state_dict(),'save_epoch':epoch}
    save_file_name='/home/minhwan/classification-frame/save.pt'
    torch.save(param_dict,save_file_name)

def LOAD(net,optim):
    net.load_state_dict(torch.load('save.pt')['model'])
    optim.load_state_dict(torch.load('save.pt')['optimizer'])
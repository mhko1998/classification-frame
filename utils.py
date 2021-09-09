import os
import sys
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'

    dist.init_process_group("gloo",rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
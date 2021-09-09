import Simrun
import DDPrun
import torch.multiprocessing as mp
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '172.27.183.200'
    os.environ['MASTER_PORT'] = '6006'
    mp.spawn(DDPrun.DDPrun, nprocs=args.gpus, args=(args,))
    
if __name__=="__main__":
    main()
import Simrun
import DDPrun
import torch.multiprocessing as mp
import os
import argparse
import random
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '172.27.183.200'
    a = random.randint(1,9999)
    os.environ['MASTER_PORT'] = str(a)
    mp.spawn(DDPrun.DDPrun, nprocs=args.gpus, args=(args,))
    
if __name__=="__main__":
    main()
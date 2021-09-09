import Simrun
import DDPrun
import torch.multiprocessing as mp
import os
import argparse
import random
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default='3,4', type=str,
                        help='the number for used gpus')
    args = parser.parse_args()

    args.world_size = len(args.gpus.split(','))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    os.environ['MASTER_ADDR'] = '172.27.183.200'
    a = random.randint(1,9999)
    os.environ['MASTER_PORT'] = str(a)
    
    mp.spawn(DDPrun.DDPrun, nprocs=args.world_size, args=(args, ))
    
if __name__=="__main__":
    main()
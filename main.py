import DDPrun
import torch.multiprocessing as mp
import os
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default='0,1', type=str,
                        help='the number for used gpus')
    parser.add_argument('-l','--load',default='True',type=lambda x: x.lower()=='true')
    parser.add_argument('--num_epochs',type=int,default=21)
    parser.add_argument('--pathdir',default='/home/minhwan/classification-frame',type=str)
    parser.add_argument('--datadir',default='/home/minhwan/KFOOD_small1/original',type=str)
    args = parser.parse_args()
    print(args)
    
    args.world_size = len(args.gpus.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    os.environ['MASTER_ADDR'] = '172.27.183.200'
    a = random.randint(1,9999)
    os.environ['MASTER_PORT'] = str(a)
    
    mp.spawn(DDPrun.DDPrun, nprocs=args.world_size, args=(args, ))
    
if __name__=="__main__":
    main()
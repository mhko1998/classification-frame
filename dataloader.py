import torch
import glob
import torchvision.transforms as transforms
import argparse
from torch.utils.data import Dataset
from PIL import Image
import argparse
import torch.utils.data.distributed as dist

class ImageDataLoader(Dataset):
    def __init__(self, dir ,images, transform):
        self.images = images
        self.transform = transform
        self.dir = dir
        self.label_dict, self.name_dict=self.__labeling__()
        
    def __labeling__(self):
        dirname = self.dir
        label_dict = dict()
        name_dict = dict()
        i= [i for i in range(len(dirname))]
        for y in range(len(dirname)):
            label = dirname[y].split('/')[-1]
            label_dict[label] = i[y]
            name_dict[i[y]]=label
        return label_dict, name_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self ,index):
        imgname = self.images[index]
        x=imgname.split('/')[-2]
        label=self.label_dict[x]
        image = Image.open(imgname)
        image=self.transform(image)
        return image, label


def data_loader():
    trans=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    trainimages= glob.glob('/home/minhwan/KFOOD_small1/original/train/*/*.jpg')
    traindir = glob.glob('/home/minhwan/KFOOD_small1/original/train/*')

    testimages=glob.glob('/home/minhwan/KFOOD_small1/original/test/*/*.jpg')
    testdir = glob.glob('/home/minhwan/KFOOD_small1/original/test/*')
    trainset=ImageDataLoader(traindir,trainimages,trans)
    testset=ImageDataLoader(testdir,testimages,trans)
    batch_size=32
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=8)
    testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=8)
    return trainloader, testloader

def DDP_data_loader(rank, world_size):
    trans=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    
    trainimages= glob.glob('/home/minhwan/KFOOD_small1/original/train/*/*.jpg')
    traindir = glob.glob('/home/minhwan/KFOOD_small1/original/train/*')

    testimages=glob.glob('/home/minhwan/KFOOD_small1/original/test/*/*.jpg')
    testdir = glob.glob('/home/minhwan/KFOOD_small1/original/test/*')
    trainset=ImageDataLoader(traindir,trainimages,trans)
    testset=ImageDataLoader(testdir,testimages,trans)
    batch_size=32
    train_sampler=dist.DistributedSampler(
        dataset=trainset,
        num_replicas=world_size,
        rank=rank
    )
    test_sampler=dist.DistributedSampler(
        dataset=testset,
        num_replicas=world_size,
        rank=rank
    )
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True,sampler=train_sampler)
    testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True,sampler=train_sampler)
    return trainloader, testloader
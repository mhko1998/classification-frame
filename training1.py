import neptune.new as neptune
from neptune.new import run
import torch
import torch.distributed as dist
from tqdm import tqdm

def train(net1,trainloader,optimizer,criterion,device,epoch):
    net1.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        inputs1, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net1(inputs1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('[%d] loss: %.3f' % (epoch + 1, running_loss /len(trainloader)))
    print(len(trainloader))

def test(net1,testloader,device):
    net1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader,0)):
            images1, labels = data[0].to(device), data[1].to(device)
            outputs = net1(images1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' %(100*correct/total))
    print(correct , total)

def DDPtraining(rank,net1,trainloader,optimizer,criterion,epoch, run):
    net1.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        inputs1, labels = data[0].to(rank), data[1].to(rank)
        optimizer.zero_grad()
        outputs = net1(inputs1).to(rank)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss=running_loss/len(trainloader)
    print('[%d] loss: %.3f' % (epoch + 1, avg_loss))
    print(len(trainloader))
    run['loss'].log(avg_loss)

def DDPtest(rank,net1, testloader, run, max):
    net1.eval()
    print(max)
    correct=0
    total=0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader,0)):
            inputs1, labels = data[0].to(rank), data[1].to(rank)
            outputs=net1(inputs1).to(rank)
            _, predicted= torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        acc=100*correct/total
        print('Accuracy of the network on the test images: %d %%'%(100*correct/total))
        print(correct ,total)
        run['acc'].log(acc)
    return acc
            
        
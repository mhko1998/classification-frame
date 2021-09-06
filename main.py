import torch
import dataloader
import getnet
import optim
import torch.nn as nn
import training1
import test1
device=torch.device('cuda:2' if torch.cuda.is_available() else'cpu')

trainloader,testloader=dataloader.data_loader()

net1 = getnet.Net()
net1 = net1.to(device)

criterion = nn.CrossEntropyLoss()

optimizer=optim.getoptim(net1)

maxi = 0
for epoch in range(21):
    training1.train(net1,trainloader,optimizer,criterion,device,epoch)

    if epoch % 5 == 0:
        test1.test(net1,testloader,criterion,device)
print('Finished Training')
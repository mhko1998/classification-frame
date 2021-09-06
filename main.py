import torch
import dataloader
import getnet
import optim
import torch.nn as nn
from tqdm import tqdm

device=torch.device('cuda:2' if torch.cuda.is_available() else'cpu')

trainloader,testloader=dataloader.data_loader()

net1 = getnet.Net1()
net1 = net1.to(device)

criterion = nn.CrossEntropyLoss()

optimizer=optim.getoptim()

maxi = 0
for epoch in range(21):
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
    
    if epoch % 5 == 0:
        net1.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader,0)):
                images1, labels = data[0].to(device), data[1].to(device)
                outputs = net1(images1)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' %(100*correct/total))
        print(correct , total)
print('Finished Training')
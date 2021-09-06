import torch
from tqdm import tqdm

def test(net1,testloader,criterion,device):
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
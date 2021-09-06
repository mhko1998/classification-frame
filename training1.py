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
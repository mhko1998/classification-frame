import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,4,kernel_size=3,stride=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4,16,padding=1,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(16,64,padding=1,kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(64,256,3,padding=1,stride=1)
        self.conv5 = nn.Conv2d(256,1024,3,stride=1)
        self.fc1 = nn.Linear(11*11*1024,2048)
        self.fc3 = nn.Linear(2048,1000)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x=self.fc3(x)
        return x

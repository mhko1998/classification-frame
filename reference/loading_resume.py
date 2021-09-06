import torch
import torch.nn as nn

## SAVE
model = nn.Conv2d(3, 64, 3, 1, 0)
optim = torch.optim.SGD(model.parameters(), lr=1e-4)
epoch = 12 

param_dict = {'model': model.state_dict(), 'optimizer': optim.state_dict(), 'save_epoch': epoch}
save_file_name = 'save.pt'
torch.save(param_dict, save_file_name)

## LOAD
model = nn.Conv2d(3, 64, 3, 1, 0)
optim = torch.optim.SGD(model.parameters(), lr=1e-4)

model.load_state_dict(torch.load('save.pt')['model'])
optim.load_state_dict(torch.load('save.pt')['optimizer'])

## RESUME





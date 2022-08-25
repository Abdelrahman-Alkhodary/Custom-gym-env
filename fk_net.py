import torch
import torch.nn as nn


class FK_Net(nn.Module):
    def __init__(self):
        super(FK_Net, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(11, 256),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256,256),
                                 nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(256,256),
                                 nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(256,3))
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
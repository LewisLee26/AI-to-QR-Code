import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, mnist=True):
      
        super(Net, self).__init__()
        if mnist:
          num_channels = 1
        else:
          num_channels = 3
          
        self.conv1 = nn.Conv2d(num_channels, 4, 5, 1)
        self.conv2 = nn.Conv2d(4, 8, 5, 1)
        if mnist:
          self.fc1 = nn.Linear(4*4*8, 10)
          self.flatten_shape = 4*4*8
        else:
          self.fc1 = nn.Linear(1250, 500)
          self.flatten_shape = 1250
   
      
    def forward(self, x, vis=False, axs=None):
        X = 0
        y = 0

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)  
        x = x.view(-1, self.flatten_shape)
        x = F.relu(self.fc1(x))


        return F.log_softmax(x, dim=1)
    

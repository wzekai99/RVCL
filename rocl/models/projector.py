import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, input_size=100, medium=300):
        super(Projector, self).__init__()

        self.linear_1 = nn.Linear(input_size, medium)
        self.linear_2 = nn.Linear(medium, input_size)
    
    def forward(self, x):
            
        output = self.linear_1(x)
        output = F.relu(output)

        output = self.linear_2(output)
        
        return output


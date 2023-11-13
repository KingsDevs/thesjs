import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim, device=self.device)
        self.key = nn.Linear(input_dim, input_dim, device=self.device)
        self.value = nn.Linear(input_dim, input_dim, device=self.device)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
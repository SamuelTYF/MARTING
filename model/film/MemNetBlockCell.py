import torch
import torch.nn.functional as F
from model.film.Attention import Attention
import torch
import torch.nn as nn

class MemNetBlockCell(nn.Module):

    def __init__(self,
        x_dim,
        y_dim,
        x_feature_dim
        ):
        super(MemNetBlockCell, self).__init__()
        self.gammas=nn.Linear(y_dim,x_dim)
        self.betas=nn.Linear(y_dim,x_dim)

        self.attention = Attention(x_feature_dim, score_function='mlp')
        self.x_linear = nn.Linear(x_feature_dim, x_feature_dim)

    # x  [BatchSize,x_dim,x_feature_dim]
    # y  [BatchSize,y_dim]
    # x' [BatchSize,x_dim,x_feature_dim]
    def forward(self, x, y):
        gamma=(self.gammas(y)+1).unsqueeze(2).expand_as(x)
        beta=self.betas(y).unsqueeze(2).expand_as(x)
        memory = self.x_linear(x)
        out, _ = self.attention(x, memory)
        out = gamma*out+beta
        memory = F.relu(out) + x
        return memory

net=MemNetBlockCell(x_dim=36,y_dim=256,x_feature_dim=256)
x=torch.ones([32,36,256])
y=torch.ones([32,256])
xp=net(x,y)
print(xp.shape)
import torch
import torch.nn.functional as F
from model.film.Attention import Attention
import torch
import torch.nn as nn

class MemNet(nn.Module):

    def __init__(self,
        embed_dim,
        output_dim=1024,
        module_dim=1024,
        hops=4,
        feature_count=36,
        feature_dim=1024
        ):
        super(MemNet, self).__init__()
        self.module_dim=module_dim
        self.hops=hops
        self.feature_count=feature_count
        self.feature_size=feature_count*feature_dim
        self.feature_linear_gamma=nn.Linear(self.feature_size,hops*module_dim)
        self.feature_linear_beta=nn.Linear(self.feature_size,hops*module_dim)

        self.gru=nn.GRU(input_size=embed_dim,hidden_size=module_dim,bidirectional=True)
        self.attention = Attention(module_dim, score_function='mlp')
        self.x_linear = nn.Linear(module_dim, module_dim)
        self.dense = nn.Linear(module_dim, output_dim)

    def forward(self, token, feature):
        x=F.softmax(feature,dim=2)
        x=x[:,0:self.feature_count,:].view([-1,self.feature_size])
        gamma=self.feature_linear_gamma(x).view([-1,self.hops,self.module_dim])+1
        beta=self.feature_linear_beta(x).view([-1,self.hops,self.module_dim])
        memory_idx = torch.sum(token != 0, dim=-1)-1
        memory_idx = memory_idx.unsqueeze(1).unsqueeze(2).expand(token.shape[0], 1, self.module_dim)
        memory = self.embed(token)
        memory,_=self.gru(memory)
        memory=memory.gather(1,memory_idx)
        for i in range(self.hops):
            x = self.x_linear(memory)
            out, _ = self.attention(memory, x)
            out = gamma[:,i:i+1]*out+beta[:,i:i+1]
            memory = F.relu(out) + memory
        memory=memory.view([-1,self.module_dim])
        result = self.dense(memory)
        return result

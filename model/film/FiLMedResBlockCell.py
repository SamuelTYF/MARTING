import torch
import torch.nn as nn
import torch.nn.functional as F
class FiLMedResBlockCell(nn.Module):
  def __init__(self,x_dim,y_dim, dropout, kernel_size):
    super(FiLMedResBlockCell, self).__init__()
    self.dropout = dropout
    self.kernel_size = kernel_size

    if self.kernel_size % 2 == 0:
      raise(NotImplementedError)

    self.gammas=nn.Linear(y_dim,x_dim)
    self.betas=nn.Linear(y_dim,x_dim)
    self.input_proj = nn.Conv1d(x_dim, x_dim, kernel_size=1)

    self.conv1 = nn.Conv1d(x_dim, x_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
    self.bn1 = nn.BatchNorm1d(x_dim, affine=False)
    if dropout > 0:
      self.drop = nn.Dropout(p=self.dropout)

    nn.init.kaiming_uniform_(self.input_proj.weight)
    nn.init.kaiming_uniform_(self.conv1.weight)

  # x  [BatchSize,x_dim,x_feature_dim]
  # y  [BatchSize,y_dim]
  # x' [BatchSize,x_dim,x_feature_dim]
  def forward(self, x, y):
    # ResBlock input projection
    x = F.relu(self.input_proj(x))
    out = x

    # ResBlock body
    out = self.conv1(out)
    out = self.bn1(out)
    # [BatchSize,x_dim,x_feature_dim]
    gamma=(self.gammas(y)+1).unsqueeze(2).expand_as(out)
    beta=self.betas(y).unsqueeze(2).expand_as(out)
    out = gamma*out + beta
    if self.dropout > 0:
      out = self.drop(out)
    out = F.relu(out)

    # ResBlock remainder
    out = x + out
    return out

net=FiLMedResBlockCell(x_dim=36,y_dim=256,dropout=0.1,kernel_size=3)
x=torch.ones([32,36,256])
y=torch.ones([32,256])
xp=net(x,y)
print(xp.shape)
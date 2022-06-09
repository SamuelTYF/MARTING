import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMGen(nn.Module):
  def __init__(self,
    hidden_dim=768,
    num_modules=4,
    module_dim=128
  ):
    super(FiLMGen, self).__init__()
    self.num_modules = num_modules
    self.module_dim = module_dim
    self.hidden_dim=hidden_dim
    self.gamma_linear = nn.Linear(self.hidden_dim, self.num_modules * self.module_dim)
    self.beta_linear = nn.Linear(self.hidden_dim, self.num_modules * self.module_dim)
    nn.init.kaiming_uniform_(self.gamma_linear.weight)
    nn.init.kaiming_uniform_(self.beta_linear.weight)

  def forward(self, x):
    gamma = self.gamma_linear(x).view(-1, self.num_modules, self.module_dim)+1
    beta = self.beta_linear(x).view(-1, self.num_modules, self.module_dim)
    return (gamma,beta)

class FiLM(nn.Module):
  def forward(self, x, gammas, betas):
    gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
    betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    return (gammas * x) + betas

class FiLMedNet(nn.Module):
  def __init__(self,
               num_modules=4,
               module_dim=128,
               module_dropout=0,
               classifier_proj_dim=512,
               classifier_downsample='maxpool',
               classifier_fc_dim=1024,
               classifier_dropout=0
               ):
    super(FiLMedNet, self).__init__()

    self.num_modules = num_modules
    self.module_dim = module_dim
    self.coords = torch.nn.Parameter(coord_map(14,14))
    stem_feature_dim = 1024 + 2
    self.stem = build_stem(stem_feature_dim, module_dim, kernel_size=3)
    self.function_modules = {}
    for fn_num in range(self.num_modules):
      mod = FiLMedResBlock(module_dim,module_dim,dropout=module_dropout,kernel_size=3)
      self.add_module(str(fn_num), mod)
      self.function_modules[fn_num] = mod
    self.classifier = build_classifier(module_dim + 2,
                                       classifier_fc_dim, classifier_proj_dim,
                                       classifier_downsample,
                                       dropout=classifier_dropout)
    for m in self.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)

  def forward(self, x, gammas,betas):
    batch_coords = self.coords.unsqueeze(0).expand(torch.Size((x.size(0), *self.coords.size())))
    x = torch.cat([x, batch_coords], 1)
    temp = self.stem(x)
    for fn_num in range(self.num_modules):
      temp = self.function_modules[fn_num](temp,gammas[:,fn_num,:], betas[:,fn_num,:], batch_coords)

    temp = torch.cat([temp, batch_coords], 1)
    out = self.classifier(temp)

    return out


class FiLMedResBlock(nn.Module):
  def __init__(self, in_dim, out_dim, dropout, kernel_size):
    super(FiLMedResBlock, self).__init__()
    self.dropout = dropout
    self.kernel_size = kernel_size

    if self.kernel_size % 2 == 0:
      raise(NotImplementedError)

    self.input_proj = nn.Conv2d(in_dim + 2, in_dim, kernel_size=1)

    self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
    self.bn1 = nn.BatchNorm2d(out_dim, affine=False)
    self.film = FiLM()
    if dropout > 0:
      self.drop = nn.Dropout2d(p=self.dropout)

    nn.init.kaiming_uniform_(self.input_proj.weight)
    nn.init.kaiming_uniform_(self.conv1.weight)

  def forward(self, x, gammas, betas, extra_channels):
    x = torch.cat([x, extra_channels], 1)
    x = F.relu(self.input_proj(x))
    out = x
    out = self.conv1(out)
    out = self.bn1(out)
    out = self.film(out, gammas, betas)
    if self.dropout > 0:
      out = self.drop(out)
    out = F.relu(out)
    out = x + out
    return out


def coord_map(m,n, start=-1, end=1):
  x_coord_row = torch.linspace(start, end, steps=n).type(torch.FloatTensor)
  y_coord_row = torch.linspace(start, end, steps=m).type(torch.FloatTensor)
  x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0).clone()
  y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0).clone()
  return torch.cat([x_coords, y_coords], 0)

def build_stem(feature_dim, module_dim,kernel_size):
  if kernel_size % 2 == 0:
    raise(NotImplementedError)
  padding = kernel_size // 2
  return nn.Sequential(
      nn.Conv2d(feature_dim, module_dim, kernel_size=kernel_size,stride=1,padding=padding),
      nn.BatchNorm2d(module_dim,affine=True),
      nn.ReLU(inplace=True)
  )

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

def build_classifier(module_C, fc_dim, proj_dim, downsample, dropout=0):
  layers = []
  prev_dim = module_C * 14 * 14
  layers.append(nn.Conv2d(module_C, proj_dim, kernel_size=1))
  layers.append(nn.BatchNorm2d(proj_dim))
  layers.append(nn.ReLU(inplace=True))
  prev_dim = proj_dim * 14 * 14
  if 'maxpool' in downsample or 'avgpool' in downsample:
    pool = nn.MaxPool2d if 'maxpool' in downsample else nn.AvgPool2d
    layers.append(pool(kernel_size=2, stride=2, padding=0))
    prev_dim = proj_dim * 7 * 7
  layers.append(Flatten())
  layers.append(nn.Linear(prev_dim, fc_dim))
  layers.append(nn.BatchNorm1d(fc_dim))
  layers.append(nn.ReLU(inplace=True))
  if dropout > 0:
    layers.append(nn.Dropout(p=dropout))
  return nn.Sequential(*layers)
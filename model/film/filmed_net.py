import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.film.FiLM import FiLM


class FiLMedNet(nn.Module):
  def __init__(self, feature_dim=(1024, 14, 14),
               stem_kernel_size=3,
               num_modules=4,
               module_num_layers=1,
               module_dim=128,
               module_dropout=0,
               module_kernel_size=3,
               classifier_proj_dim=512,
               classifier_downsample='maxpool2',
               classifier_fc_layers=(1024,),
               classifier_dropout=0
               ):
    super(FiLMedNet, self).__init__()

    self.num_modules = num_modules
    self.module_num_layers = module_num_layers
    self.module_dim = module_dim

    # Initialize helper variables
    self.coords = coord_map((feature_dim[1], feature_dim[2])).cuda()
    
    # Initialize stem
    stem_feature_dim = feature_dim[0] + 2
    self.stem = build_stem(stem_feature_dim, module_dim, kernel_size=stem_kernel_size)

    # Initialize FiLMed network body
    self.function_modules = {}
    for fn_num in range(self.num_modules):
      mod = FiLMedResBlock(module_dim,module_dim,
                      dropout=module_dropout,
                      kernel_size=module_kernel_size
                      )
      self.add_module(str(fn_num), mod)
      self.function_modules[fn_num] = mod

    # Initialize output classifier
    self.classifier = build_classifier(module_dim + 2, feature_dim[1], feature_dim[2],
                                       classifier_fc_layers, classifier_proj_dim,
                                       classifier_downsample,
                                       dropout=classifier_dropout)
    for m in self.modules():
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)

  def forward(self, x, film):
    # Initialize forward pass and externally viewable activations
    gammas, betas = torch.split(film[:,:,:2*self.module_dim], self.module_dim, dim=-1)
    batch_coords = self.coords.unsqueeze(0).expand(torch.Size((x.size(0), *self.coords.size()))).clone()
    x = torch.cat([x, batch_coords], 1)
    module_inputs = self.stem(x)
    for fn_num in range(self.num_modules):
      layer_output = self.function_modules[fn_num](module_inputs,gammas[:,fn_num,:], betas[:,fn_num,:], batch_coords)
      if fn_num == (self.num_modules - 1):
        final_module_output = layer_output
      else:
        module_inputs=layer_output

    final_module_output = torch.cat([final_module_output, batch_coords], 1)
    out = self.classifier(final_module_output)

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
    # ResBlock input projection
    x = torch.cat([x, extra_channels], 1)
    x = F.relu(self.input_proj(x))
    out = x

    # ResBlock body
    out = self.conv1(out)
    out = self.bn1(out)
    out = self.film(out, gammas, betas)
    if self.dropout > 0:
      out = self.drop(out)
    out = F.relu(out)

    # ResBlock remainder
    out = x + out
    return out


def coord_map(shape, start=-1, end=1):
  """
  Gives, a 2d shape tuple, returns two mxn coordinate maps,
  Ranging min-max in the x and y directions, respectively.
  FASTRCNN
  
  """
  m, n = shape
  x_coord_row = torch.linspace(start, end, steps=n).type(torch.FloatTensor)
  y_coord_row = torch.linspace(start, end, steps=m).type(torch.FloatTensor)
  x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0).clone()
  y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0).clone()
  return torch.cat([x_coords, y_coords], 0)

def build_stem(feature_dim, module_dim,kernel_size):
  layers = []
  prev_dim = feature_dim
  if kernel_size % 2 == 0:
    raise(NotImplementedError)
  padding = kernel_size // 2
  layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=kernel_size,stride=1,padding=padding))
  layers.append(nn.BatchNorm2d(module_dim,affine=True))
  layers.append(nn.ReLU(inplace=True))
  return nn.Sequential(*layers)

class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

def build_classifier(module_C, module_H, module_W,
                     fc_dims, proj_dim, downsample='maxpool2',
                     dropout=0):
  layers = []
  prev_dim = module_C * module_H * module_W
  layers.append(nn.Conv2d(module_C, proj_dim, kernel_size=1))
  layers.append(nn.BatchNorm2d(proj_dim))
  layers.append(nn.ReLU(inplace=True))
  prev_dim = proj_dim * module_H * module_W
  if 'maxpool' in downsample or 'avgpool' in downsample:
    pool = nn.MaxPool2d if 'maxpool' in downsample else nn.AvgPool2d
    if 'full' in downsample:
      if module_H != module_W:
        assert(NotImplementedError)
      pool_size = module_H
    else:
      pool_size = int(downsample[-1])
    # Note: Potentially sub-optimal padding for non-perfectly aligned pooling
    padding = 0 if ((module_H % pool_size == 0) and (module_W % pool_size == 0)) else 1
    layers.append(pool(kernel_size=pool_size, stride=pool_size, padding=padding))
    prev_dim = proj_dim * math.ceil(module_H / pool_size) * math.ceil(module_W / pool_size)
  layers.append(Flatten())
  for next_dim in fc_dims:
    layers.append(nn.Linear(prev_dim, next_dim))
    layers.append(nn.BatchNorm1d(next_dim))
    layers.append(nn.ReLU(inplace=True))
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
    prev_dim = next_dim
  return nn.Sequential(*layers)
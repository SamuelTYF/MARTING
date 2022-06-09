#!/usr/bin/env python3

import torch
import torch.nn as nn

class FiLMGen(nn.Module):
  def __init__(self,
    null_token=0,
    encoder_vocab_size=100,
    wordvec_dim=200,
    hidden_dim=512,
    rnn_num_layers=1,
    rnn_dropout=0,
    num_modules=4,
    module_num_layers=1,
    module_dim=128
  ):
    super(FiLMGen, self).__init__()
    self.num_modules = num_modules
    self.module_num_layers = module_num_layers
    self.module_dim = module_dim
    self.NULL = null_token
    self.hidden_dim=hidden_dim
    self.rnn_num_layers=rnn_num_layers
    self.cond_feat_size = 2 * self.module_dim * self.module_num_layers  # FiLM params per ResBlock

    self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
    self.encoder_rnn =nn.GRU(wordvec_dim, self.hidden_dim, self.rnn_num_layers, dropout=rnn_dropout,batch_first=True)
    self.decoder_linear = nn.Linear(self.hidden_dim, self.num_modules * self.cond_feat_size)

    self.gammas = []
    for i in range(self.module_num_layers):
      self.gammas.append(slice(i * (2 * self.module_dim), i * (2 * self.module_dim) + self.module_dim))

    nn.init.kaiming_uniform_(self.decoder_linear.weight)

  def forward(self, x):
    idx=torch.sum(x!=self.NULL,dim=-1).type_as(x.data)-1
    embed = self.encoder_embed(x)
    h0 = torch.zeros(self.rnn_num_layers, x.shape[0], self.hidden_dim).type_as(embed.data)
    out, _ = self.encoder_rnn(embed, h0)
    # Pull out the hidden state for the last non-null value in each input
    idx = idx.unsqueeze(1).unsqueeze(2).expand(x.shape[0], 1, self.hidden_dim)
    encoded=out.gather(1, idx).view(x.shape[0], self.hidden_dim)
    film = self.decoder_linear(encoded).view(x.shape[0], self.num_modules, self.cond_feat_size)
    for i in range(self.module_num_layers):
      film[:,:,self.gammas[i]] = film[:,:,self.gammas[i]] + 1
    return film
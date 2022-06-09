import torch.nn as nn
from model.TRAR.layer_norm import LayerNorm
import torch
import torch.nn.functional as F
from model.TRAR.trar import AttFlat

class cls_layer_img(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_img, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, lang_feat, img_feat):
        proj_feat = self.proj_norm(img_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat

class cls_layer_txt(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_txt, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, lang_feat, img_feat):
        proj_feat = self.proj_norm(lang_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat

class cls_layer_both(nn.Module):
    def __init__(self,  input_dim, output_dim):
        super(cls_layer_both, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, lang_feat, img_feat):
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat

class RoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(RoutingBlock, self).__init__()
        self.pooling = pooling
        self.mlp_1 = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )
        # if pooling == 'attention':
        #     self.pool_1 = AttFlat(in_channel,in_channel,in_channel)
        #     self.pool_2 = AttFlat(in_channel,in_channel,in_channel)
        # elif pooling == 'avg':
        #     self.pool_1 = nn.AdaptiveAvgPool1d(1)
        #     self.pool_2 = nn.AdaptiveAvgPool1d(1)
        # elif pooling == 'fc':
        #     self.pool_1 = nn.Linear(in_channel, 1)
        #     self.pool_2 = nn.Linear(in_channel, 1)
        
    def forward(self, x, y):
        logits_1 = self.mlp_1(x) # (4,4)
        logits_2 = self.mlp_2(y) # (4,4)
        # if self.pooling == 'attention':
        #     x = self.pool_1(x, x_mask=self.make_mask(x)) #(4, 768)
        #     logits_1 = self.mlp_1(x.squeeze(-1)) # (4,4)
        #     y = self.pool_2(y, y_mask=self.make_mask(y)) #(4, 768)
        #     logits_2 = self.mlp_2(y.squeeze(-1)) # (4,4)
        # elif self.pooling == 'avg':
        #     x = x.transpose(1, 2)
        #     x = self.pool_1(x)
        #     logits_1 = self.mlp_1(x.squeeze(-1))
        #     y = y.transpose(1, 2)
        #     y = self.pool_2(y)
        #     logits_2 = self.mlp_2(y.squeeze(-1))
        # elif self.pooling == 'fc':
        #     mask = self.make_mask(x).squeeze().unsqueeze(2)
        #     scores = self.pool_1(x)
        #     scores = scores.masked_fill(mask, -1e9)
        #     scores = F.softmax(scores, dim=1)
        #     _x = x.mul(scores)
        #     x = torch.sum(_x, dim=1)
        #     logits_1 = self.mlp_1(x)

        #     mask2 = self.make_mask(y).squeeze().unsqueeze(2)
        #     scores2 = self.pool_2(y)
        #     scores2 = scores2.masked_fill(mask2, -1e9)
        #     scores2 = F.softmax(scores2, dim=1)
        #     _y = y.mul(scores2)
        #     y = torch.sum(_y, dim=1)
        #     logits_2 = self.mlp_2(y)
        logits = torch.concat([logits_1, logits_2], dim=-1)
        alpha = F.softmax(logits, dim=-1)  #
        return alpha

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

class cls_layer_both_routing(nn.Module):
    def __init__(self,  input_dim, output_dim, pooling):
        super(cls_layer_both_routing, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        self.routing = RoutingBlock(input_dim, 1, pooling)
        
    def forward(self, lang_feat, img_feat):
        alpha = self.routing(lang_feat, img_feat)
        proj_feat = torch.mul(lang_feat, alpha[:, 0:1]) + torch.mul(img_feat, alpha[:, 1:2])
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat



class cls_layer_both_concat(nn.Module):
    def __init__(self,  input_dim, output_dim):
        super(cls_layer_both_concat, self).__init__()
        self.proj_norm = LayerNorm(input_dim * 2)
        self.proj = nn.Linear(input_dim * 2, output_dim)
        
    def forward(self, lang_feat, img_feat):
        proj_feat = torch.concat([lang_feat, img_feat], dim=-1)
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat
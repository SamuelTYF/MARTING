from model.TRAR.fc import MLP
from model.TRAR.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
class AttFlat(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, glimpses=1, dropout_r=0):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = MLP(
            input_dim=in_channel,
            hidden_dim=hidden_channel,
            output_dim=glimpses,
            dropout=dropout_r,
            activation="ReLU"
        )

        self.linear_merge = nn.Linear(
            in_channel * glimpses,
            out_channel
        )
        self.norm = LayerNorm(out_channel)

    def forward(self, x, x_mask):
        att = self.mlp(x) # (bs, grid_num, dim) ->(bs, grid_num, 1)

        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9) # (bs, grid_num, 1)
        att = F.softmax(att, dim=1) # # (bs, grid_num, 1)

        att_list = [torch.sum(att[:, :, i: i + 1] * x, dim=1) for i in range(self.glimpses)] # [(4, 768)]

        x_atted = torch.cat(att_list, dim=1) # (4, 768)
        x_atted = self.linear_merge(x_atted) # (4, 768)
        x_atted = self.norm(x_atted)
        return x_atted

class SoftRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(SoftRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'attention':
            self.pool = AttFlat(in_channel,in_channel,in_channel)
        elif pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'attention':
            x = self.pool(x, x_mask=self.make_mask(x)) #(4, 768)
            logits = self.mlp(x.squeeze(-1)) # (4,4)
        elif self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze().unsqueeze(2)
            scores = self.pool(x)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)
            
        alpha = F.softmax(logits, dim=-1)  #
        return alpha

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class HardRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(HardRoutingBlock, self).__init__()
        self.pooling = pooling

        if pooling == 'attention':
            self.pool = AttFlat(in_channel)
        elif pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'attention':
            x = self.pool(x, x_mask=self.make_mask(x))
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze().unsqueeze(2)
            scores = self.pool(x)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)

        alpha = self.gumbel_softmax(logits, -1, tau)
        return alpha

    def gumbel_softmax(self, logits, dim=-1, temperature=0.1):
        '''
        Use this to replace argmax
        My input is probability distribution, multiply by 10 to get a value like logits' outputs.
        '''
        gumbels = -torch.empty_like(logits).exponential_().log()
        logits = (logits.log_softmax(dim=dim) + gumbels) / temperature
        return F.softmax(logits, dim=dim)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_k = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_q = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_merge = nn.Linear(opt["hidden_size"], opt["hidden_size"])

        self.dropout = nn.Dropout(opt["dropout"])

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt["hidden_size"]
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# -------------------------------------
# ---- Dynmaic Span Self-Attention ----
# -------------------------------------

class SARoutingBlock(nn.Module):
    """
    Self-Attention Routing Block
    """

    def __init__(self, opt):
        super(SARoutingBlock, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_k = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_q = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_merge = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        if opt["routing"] == 'hard':
            self.routing_block = HardRoutingBlock(opt["hidden_size"], opt["orders"], opt["pooling"])
        elif opt["routing"] == 'soft':
            self.routing_block = SoftRoutingBlock(opt["hidden_size"], opt["orders"], opt["pooling"])

        self.dropout = nn.Dropout(opt["dropout"])

    def forward(self, v, k, q, masks, tau, training):
        n_batches = q.size(0)
        x = v

        alphas = self.routing_block(x, tau, masks) # (bs, 4)

        if self.opt["BINARIZE"]:
            if not training:
                alphas = self.argmax_binarize(alphas)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2) # (bs, 4, 49, 192)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2) # (bs, 4, 49, 192)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2) # (bs, 4, 49, 192)

        att_list = self.routing_att(v, k, q, masks) # (bs, order_num, head_num, grid_num, grid_num) (bs, 4, 4, 49, 49)
        att_map = torch.einsum('bl,blcnm->bcnm', alphas, att_list) # (bs, 4), (bs, 4, 4, 49, 49) - > (bs, 4, 49, 49)

        atted = torch.matmul(att_map, v) # (bs, 4, 49, [49]) * (bs, 4, [49],192) - > (bs, 4, 49, 192) mul [49, 49]*[49, 192], 

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt["hidden_size"]
        ) # (bs, 49, 768)

        atted = self.linear_merge(atted) # (bs, 4, 768)

        return atted

    def routing_att(self, value, key, query, masks):
        d_k = query.size(-1) # masks [[bs, 1, 1, 49], [bs, 1, 49, 49], [bs, 1, 49, 49], [bs, 1, 49, 49]]
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k) # (bs, 4, 49, 49) (2, 4, 360, 49)
        # k q v [4, 4, 49, 192] key (2, 4, 49, 192) query [2, 4, 360, 192]
        for i in range(len(masks)):
            mask = masks[i] # (bs, 1, 49, 49)
            scores_temp = scores.masked_fill(mask, -1e9)
            att_map = F.softmax(scores_temp, dim=-1)
            att_map = self.dropout(att_map)
            if i == 0:
                att_list = att_map.unsqueeze(1) # (bs, 1, 4, 49, 49)
            else:
                att_list = torch.cat((att_list, att_map.unsqueeze(1)), 1)  # (bs, 2, 4, 49, 49) -> (bs, 3, 4, 49, 49)

        return att_list

    def argmax_binarize(self, alphas):
        n = alphas.size()[0]
        out = torch.zeros_like(alphas)
        indexes = alphas.argmax(-1)
        out[torch.arange(n), indexes] = 1
        return out
    
# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()

        self.mlp = MLP(
            input_dim=opt["hidden_size"],
            hidden_dim=opt["ffn_size"],
            output_dim=opt["hidden_size"],
            dropout=opt["dropout"],
            activation="ReLU"
        )

    def forward(self, x):
        return self.mlp(x)


# -----------------------------
# ---- Transformer Encoder ----
# -----------------------------

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.mhatt = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt["dropout"])
        self.norm1 = LayerNorm(opt["hidden_size"])

        self.dropout2 = nn.Dropout(opt["dropout"])
        self.norm2 = LayerNorm(opt["hidden_size"])

    def forward(self, y, y_mask): # y (64, 40, 512) y_mask (64, 1, 1, 40)
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        )) # (64, 40, 512)

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y

# -----------------------------
# ---- Transformer Encoder routing----
# -----------------------------

class Encoder_TRAR(nn.Module):
    def __init__(self, opt):
        super(Encoder_TRAR, self).__init__()
        self.mhatt1 = SARoutingBlock(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt["dropout"])
        self.norm1 = LayerNorm(opt["hidden_size"])

        self.dropout2 = nn.Dropout(opt["dropout"])
        self.norm2 = LayerNorm(opt["hidden_size"])

    def forward(self, y, y_masks, tau, training): # y (64, 40, 512) y_mask (64, 1, 1, 40)
        y = self.norm1(y + self.dropout1(
            self.mhatt1(v=y, k=y, q=y, masks=y_masks, tau=tau, training=training)
        )) # (64, 49, 512) # (bs, 49, 768)


        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y

# ---------------------------------
# ---- Multimodal TRAR Decoder ----
# ---------------------------------
class TRAR(nn.Module):
    def __init__(self, opt):
        super(TRAR, self).__init__()

        self.mhatt1 = SARoutingBlock(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt["dropout"])
        self.norm1 = LayerNorm(opt["hidden_size"])

        self.dropout2 = nn.Dropout(opt["dropout"])
        self.norm2 = LayerNorm(opt["hidden_size"])

        self.dropout3 = nn.Dropout(opt["dropout"])
        self.norm3 = LayerNorm(opt["hidden_size"])

    def forward(self, x, y, x_masks, y_mask, tau, training): # x (64, 49, 512) y

        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, masks=x_masks, tau=tau, training=training)
        )) # (64, 49, 512) # (bs, 49, 768)

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


class Bi_multiTRAR_SA_block(nn.Module):
    def __init__(self, opt):
        super(Bi_multiTRAR_SA_block, self).__init__()

        self.mhatt1 = SARoutingBlock(opt)
        self.mhatt2 = SARoutingBlock(opt)
        self.mhatt3 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt["dropout"])
        self.norm1 = LayerNorm(opt["hidden_size"])

        self.dropout2 = nn.Dropout(opt["dropout"])
        self.norm2 = LayerNorm(opt["hidden_size"])

        self.dropout3 = nn.Dropout(opt["dropout"])
        self.norm3 = LayerNorm(opt["hidden_size"])

        self.dropout4 = nn.Dropout(opt["dropout"])
        self.norm4 = LayerNorm(opt["hidden_size"])

    def forward(self, x, y, y_masks, x_mask, x_masks, y_mask, tau, training): # x (64, 49, 512) y

        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=y, k=y, q=x, masks=y_masks, tau=tau, training=training)
        )) # (64, 49, 512) # (bs, 49, 768)

        y = self.norm2(y + self.dropout2(
            self.mhatt2(v=x, k=x, q=y, masks=x_masks, tau=tau, training=training)
        )) # (64, 360, 512) # (bs, 360, 768)

        x_y = torch.concat([x, y], dim = 1) # (bs, max_len + grid_num, dim)
        x_y_mask = torch.concat([x_mask, y_mask], dim = -1) # (bs, 1, 1, max_len + grid_num)
        x_y = self.norm3(x_y + self.dropout3(
            self.mhatt3(v=x_y, k=x_y, q=x_y, mask=x_y_mask)
        ))

        x_y = self.norm4(x_y + self.dropout4(
            self.ffn(x_y)
        ))
        x_y_split = torch.split(x_y, [x.size(1), y.size(1)], dim=1)
        return x_y_split[0], x_y_split[1] 

class multiTRAR_SA_block(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_block, self).__init__()

        self.mhatt1 = SARoutingBlock(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt["dropout"])
        self.norm1 = LayerNorm(opt["hidden_size"])

        self.dropout2 = nn.Dropout(opt["dropout"])
        self.norm2 = LayerNorm(opt["hidden_size"])

        self.dropout3 = nn.Dropout(opt["dropout"])
        self.norm3 = LayerNorm(opt["hidden_size"])

    def forward(self, x, y, y_masks, x_mask, tau, training): # x (64, 49, 512) y

        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=y, k=y, q=x, masks=y_masks, tau=tau, training=training)
        )) # (64, 49, 512) # (bs, 49, 768)

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x



class SA_multiTRAR_block(nn.Module):
    def __init__(self, opt):
        super(SA_multiTRAR_block, self).__init__()

        self.mhatt1 = SARoutingBlock(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt["dropout"])
        self.norm1 = LayerNorm(opt["hidden_size"])

        self.dropout2 = nn.Dropout(opt["dropout"])
        self.norm2 = LayerNorm(opt["hidden_size"])

        self.dropout3 = nn.Dropout(opt["dropout"])
        self.norm3 = LayerNorm(opt["hidden_size"])

    def forward(self, x, y, y_masks, x_mask, tau, training): # x (64, 49, 512) y

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=y, k=y, q=x, masks=y_masks, tau=tau, training=training)
        )) # (64, 49, 512) # (bs, 49, 768)

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

# ----------------------------------------
# ---- Encoder-Decoder with TRAR Block----
# ----------------------------------------
class TRAR_ED(nn.Module):
    def __init__(self, opt):
        super(TRAR_ED, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([TRAR(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):

        y_masks = [y_mask, y_mask, y_mask, y_mask] # len 4 (64, 1, 1, 40)
        for enc in self.enc_list: # len 6
            x = enc(x, x_mask) # (64, 49, 512)
        for dec in self.dec_list:
            y = dec(y, x, y_masks, x_mask, self.tau) # (64, 40, 512)

        return y, x

    def set_tau(self, tau):
        self.tau = tau

# ----------------------------------------
# ---- multiTRAR Block -> SA text guide----
# ----------------------------------------
class multiTRAR_SA_ED_img_guide(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_ED_img_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        y_masks = getMasks_txt_multimodal(y_mask, self.opt) # [(bs, 1, 1, grid_num), (bs, 1, max_len, grid_num), (bs, 1, max_len, grid_num), (bs, 1, max_len, grid_num)]
        for enc in self.enc_list:
            y = enc(y, y_mask) # (64, 40, 512)
        for dec in self.dec_list:
            x = dec(x, y, y_masks, x_mask, self.tau, self.training)
        
        return y, x

    def set_tau(self, tau):
        self.tau = tau


class multiTRAR_SA_routing_on_span_ED_img_guide(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_routing_on_span_ED_img_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        y_masks = getMasks_txt_multimodal_with_all_img(y_mask, self.opt) # [(bs, 1, 1, grid_num), (bs, 1, max_len, grid_num), (bs, 1, max_len, grid_num), (bs, 1, max_len, grid_num)]
        for enc in self.enc_list:
            y = enc(y, y_mask) # (64, 40, 512)
        for dec in self.dec_list:
            x = dec(x, y, y_masks, x_mask, self.tau, self.training)
        
        return y, x

    def set_tau(self, tau):
        self.tau = tau


# ----------------------------------------
# ---- multiTRAR Block -> SA img guide----
# ----------------------------------------
class multiTRAR_SA_ED_text_guide(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_ED_text_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        x_masks = getMasks_img_multimodal(x_mask, self.opt)
        for enc in self.enc_list:
            x = enc(x, x_mask) # (64, 40, 512)
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            y = dec(y, x, x_masks, y_mask, self.tau, self.training) # (4, 360, 768)
        return y, x

    def set_tau(self, tau):
        self.tau = tau


# ----------------------------------------
# ---- multiTRAR Block -> SA img guide----
# ----------------------------------------
class multiTRAR_SA_routing_on_span_ED_text_guide(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_routing_on_span_ED_text_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        x_masks = getMasks_img_multimodal_with_all_text(x_mask, self.opt) # list 1+ 50
        
        for enc in self.enc_list:
            x = enc(x, x_mask) # (64, 40, 512)
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            y = dec(y, x, x_masks, y_mask, self.tau, self.training) # (4, 360, 768)
        return y, x

    def set_tau(self, tau):
        self.tau = tau



# ----------------------------------------
# ---- multiTRAR Block -> SA img guide TRAR encoder----
# ----------------------------------------
class multiTRAR_SA_TRARED_text_guide(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_TRARED_text_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder_TRAR(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        x_masks_uni = getMasks(x_mask, self.opt)
        x_masks_multi = getMasks_img_multimodal(x_mask, self.opt)
        for enc in self.enc_list:
            x = enc(x, x_masks_uni, self.tau, self.training) # (64, 40, 512)
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            y = dec(y, x, x_masks_multi, y_mask, self.tau, self.training) # (4, 360, 768)
        return y, x

    def set_tau(self, tau):
        self.tau = tau

# ----------------------------------------
# ---- multiTRAR Block -> SA text guide----
# ----------------------------------------
class multiTRAR_SA_TRARED_img_guide(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_TRARED_img_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder_TRAR(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        y_masks_uni = getMasks_txt(y_mask, self.opt) # [(bs, 1, 1, grid_num), (bs, 1, max_len, grid_num), (bs, 1, max_len, grid_num), (bs, 1, max_len, grid_num)]
        y_masks_multi = getMasks_txt_multimodal(y_mask, self.opt)
        for enc in self.enc_list:
            y = enc(y, y_masks_uni, self.tau, self.training) # (64, 40, 512)
        for dec in self.dec_list:
            x = dec(x, y, y_masks_multi, x_mask, self.tau, self.training)
        
        return y, x

    def set_tau(self, tau):
        self.tau = tau



# ----------------------------------------
# ---- SA -> multiTRAR Block text guide----
# ----------------------------------------
class SA_multiTRAR_ED_img_guide(nn.Module):
    def __init__(self, opt):
        super(SA_multiTRAR_ED_img_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([SA_multiTRAR_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        y_masks = getMasks_txt_multimodal(y_mask, self.opt) # [(bs, 1, 1, max_len), (bs, 1, grid_num, max_len), (bs, 1, grid_num, max_len), (bs, 1, grid_num, max_len)]
        for enc in self.enc_list:
            y = enc(y, y_mask) # (64, 40, 512)
        for dec in self.dec_list:
            x = dec(x, y, y_masks, x_mask, self.tau, self.training)
        
        return y, x

    def set_tau(self, tau):
        self.tau = tau

# ----------------------------------------
# ---- SA -> multiTRAR Block img guide----
# ----------------------------------------
class SA_multiTRAR_ED_text_guide(nn.Module):
    def __init__(self, opt):
        super(SA_multiTRAR_ED_text_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([SA_multiTRAR_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        x_masks = getMasks_img_multimodal(x_mask, self.opt)
        for enc in self.enc_list:
            x = enc(x, x_mask) # (64, 40, 512)
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            y = dec(y, x, x_masks, y_mask, self.tau, self.training) # (4, 360, 768)
        return y, x

    def set_tau(self, tau):
        self.tau = tau

# --------------------------------
# ---- img Local Window Generator ----
# --------------------------------
def getImgMasks(scale=16, order=2):
    """
    :param scale: Feature Map Scale
    :param order: Local Window Size, e.g., order=2 equals to windows size (5, 5)
    :return: masks = (scale**2, scale**2)
    """
    masks = []
    _scale = scale
    assert order < _scale, 'order size be smaller than feature map scale'

    for i in range(_scale):
        for j in range(_scale):
            mask = np.ones([_scale, _scale], dtype=np.float32)
            for x in range(i - order, i + order + 1, 1):
                for y in range(j - order, j + order + 1, 1):
                    if (0 <= x < _scale) and (0 <= y < _scale):
                        mask[x][y] = 0
            mask = np.reshape(mask, [_scale * _scale])
            masks.append(mask)
    masks = np.array(masks)
    masks = np.asarray(masks, dtype=np.bool) # 0, 1 -> False True (True mask)
    return masks

# --------------------------------
# ---- Text Local Window Generator ----
# --------------------------------
def getTextMasks(text_len, order=2):
    """
    :param scale: Feature Map Scale
    :param order: Local Window Size, e.g., order=2 equals to windows size (1, 5)
    :return: masks = (1, text_len)
    """
    masks = []
    _scale = text_len
    assert 2 * order + 1 < _scale, 'order size be smaller than feature map scale'

    for i in range(_scale):
        mask = np.ones([_scale], dtype=np.float32)
        for x in range(i - order, i + order + 1, 1):
            if (0 <= x < _scale):
                mask[x] = 0
        masks.append(mask)
    masks = np.array(masks)
    masks = np.asarray(masks, dtype=np.bool) # 0, 1 -> False True (True mask)
    return masks # (text_len, text_len)



def getMasks(x_mask, __C):
    mask_list = [] # x_mask [64, 1, 1, 49]
    ORDERS = __C["ORDERS"]
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask = torch.from_numpy(getImgMasks(__C["IMG_SCALE"], order)).byte().to(x_mask.device) # (49, 49)
            mask = torch.logical_or(x_mask, mask) # (64, 1, 49, 49)
            mask_list.append(mask)
    return mask_list

def getMasks_txt(x_mask, __C):
    mask_list = [] # x_mask [64, 1, 1, 360]
    ORDERS = __C["ORDERS"]
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask = torch.from_numpy(getTextMasks(__C["len"], order)).byte().to(x_mask.device) # (360, 360)
            mask = torch.logical_or(x_mask, mask) # (64, 1, 360, 360)
            mask_list.append(mask)
    return mask_list

def getMasks_img_multimodal(x_mask, __C):
    mask_list = [] # x_mask [64, 1, 1, 49]
    ORDERS = __C["ORDERS"]
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask_img = torch.from_numpy(getImgMasks(__C["IMG_SCALE"], order)).byte().to(x_mask.device) # (49, 49)
            mask = torch.concat([mask_img]*(__C["len"]//(__C["IMG_SCALE"]*__C["IMG_SCALE"])), dim=0) 
            mask = torch.concat([mask, mask_img[:(__C["len"]%(__C["IMG_SCALE"]*__C["IMG_SCALE"])),:]])
            mask = torch.logical_or(x_mask, mask) # (64, 1, max_len, grid_num)
            mask_list.append(mask)
    return mask_list 

def getMasks_img_multimodal_with_all_text(x_mask, __C):
    mask_list = [] # x_mask [64, 1, 1, 49]
    ORDERS = __C["ORDERS"]
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask_img = torch.from_numpy(getImgMasks(__C["IMG_SCALE"], order)).byte().to(x_mask.device) # (49, 49)
            for i in range(mask_img.shape[0]):
                mask_region = mask_img[i, :]
                mask = mask_region.unsqueeze(0).repeat(__C["len"], 1)
                mask = torch.logical_or(x_mask, mask)
                mask_list.append(mask)
    return mask_list


def getMasks_txt_multimodal(x_mask, __C):
    mask_list = [] # x_mask [64, 1, 1, 360]
    ORDERS = __C["ORDERS"]
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask_txt = torch.from_numpy(getTextMasks(__C["len"], order)).byte().to(x_mask.device) # (360, 360)
            mask_split = torch.chunk(mask_txt[:(__C["len"] // (__C["IMG_SCALE"]*__C["IMG_SCALE"])) * (__C["IMG_SCALE"]*__C["IMG_SCALE"])], __C["len"] // (__C["IMG_SCALE"]*__C["IMG_SCALE"]), dim=0)
            mask = torch.concat([mask_txt[(__C["len"] // (__C["IMG_SCALE"]*__C["IMG_SCALE"])) * (__C["IMG_SCALE"]*__C["IMG_SCALE"]):], torch.zeros((__C["IMG_SCALE"]*__C["IMG_SCALE"]) - __C["len"]%(__C["IMG_SCALE"]*__C["IMG_SCALE"]), __C["len"]).byte().to(x_mask.device)], dim=0)
            for mask_ in mask_split:
                mask = torch.logical_or(mask_, mask)
            mask = torch.logical_or(x_mask, mask) # (64, 1, 360, 360)
            mask_list.append(mask)
    return mask_list # (64, 1, grid_num, max_len)


def getMasks_txt_multimodal_with_all_img(x_mask, __C):
    mask_list = [] # x_mask [64, 1, 1, 360]
    ORDERS = __C["ORDERS"]
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask_txt = torch.from_numpy(getTextMasks(__C["len"], order)).byte().to(x_mask.device) # (360, 360)
            for i in range(mask_txt.shape[0]):
                mask_region = mask_txt[i, :]
                mask = mask_region.unsqueeze(0).repeat((__C["IMG_SCALE"]*__C["IMG_SCALE"]), 1)
                mask = torch.logical_or(x_mask, mask)
                mask_list.append(mask)
    return mask_list # (64, 1, grid_num, max_len)

# ----------------------------------------
# ---- Encoder-Decoder with TRAR Block----
# ----------------------------------------
class TRAR_ED_txt_guide(nn.Module):
    def __init__(self, opt):
        super(TRAR_ED_txt_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([TRAR(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        # y_masks = [y_mask, y_mask, y_mask, y_mask] # len 4 (64, 1, 1, 40)
        # for enc in self.enc_list: # len 6
        #     x = enc(x, x_mask) # (64, 49, 512)
        # for dec in self.dec_list:
        #     y = dec(y, x, y_masks, x_mask, self.tau) # (64, 40, 512)

        # return y, x
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        y_masks = getMasks_txt(y_mask, self.opt)
        for enc in self.enc_list:
            x = enc(x, x_mask) # (64, 40, 512)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            y = dec(y, x, y_masks, x_mask, self.tau, self.training) # (4, 360, 768)

        return y, x

    def set_tau(self, tau):
        self.tau = tau

class TRAR_ED_img_guide(nn.Module):
    def __init__(self, opt):
        super(TRAR_ED_img_guide, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_list = nn.ModuleList([TRAR(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        x_masks = getMasks(x_mask, self.opt)
        for enc in self.enc_list:
            y = enc(y, y_mask) # (64, 40, 512)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_masks, y_mask, self.tau, self.training)

        return y, x

    def set_tau(self, tau):
        self.tau = tau

class BiTRAR_crossSA_ED(nn.Module):
    def __init__(self, opt):
        super(BiTRAR_crossSA_ED, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.enc_txt_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_img_list = nn.ModuleList([TRAR(opt) for _ in range(opt["layer"])])
        self.enc_img_list = nn.ModuleList([Encoder(opt) for _ in range(opt["layer"])])
        self.dec_txt_list = nn.ModuleList([TRAR(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        x_masks = getMasks(x_mask, self.opt)
        y_masks = getMasks_txt(y_mask, self.opt)
        for enc_txt in self.enc_txt_list:
            y_enc = enc_txt(y, y_mask) # (64, 40, 512)
        for enc_img in self.enc_img_list:
            x_enc = enc_img(x, x_mask) # (64, 40, 512)
        for i in range(self.opt["layer"]):
            x = self.dec_img_list[i](x, y_enc, x_masks, y_mask, self.tau, self.training)
            y = self.dec_txt_list[i](y, x_enc, y_masks, x_mask, self.tau, self.training)
        return y, x

    def set_tau(self, tau):
        self.tau = tau

class Bi_multiTRAR_SA_ED(nn.Module):
    def __init__(self, opt):
        super(Bi_multiTRAR_SA_ED, self).__init__()
        self.opt = opt
        self.tau = opt["tau_max"]
        self.dec_list = nn.ModuleList([Bi_multiTRAR_SA_block(opt) for _ in range(opt["layer"])])

    def forward(self, y, x, y_mask, x_mask):
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        x_masks = getMasks_img_multimodal(x_mask, self.opt) # [(bs, 1, 1, grid_num), (bs, 1, max_len, grid_num), (bs, 1, max_len, grid_num), (bs, 1, max_len, grid_num)]
        y_masks = getMasks_txt_multimodal(y_mask, self.opt) # [(bs, 1, 1, max_len), (bs, 1, grid_num, max_len), (bs, 1, grid_num, max_len), (bs, 1, grid_num, max_len)]
        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x, y = dec(x, y, y_masks, x_mask, x_masks, y_mask, self.tau, self.training)
    
        return y, x

    def set_tau(self, tau):
        self.tau = tau
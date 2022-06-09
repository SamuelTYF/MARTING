import torch
import math
import timm
import model
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a * (x - mean) / (std + self.eps) + self.b


class TRAR_Twostream_concat(torch.nn.Module):
  # define model elements
    def __init__(self,bertl_text,vit, opt):
        super(TRAR_Twostream_concat, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.vit=vit
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        opt["backbone"] = "text_guide"
        self.trar_text = model.TRAR.multiTRAR_SA(opt)
        opt["backbone"] = "img_guide"
        self.trar_img = model.TRAR.multiTRAR_SA(opt)
        self.sigm = torch.nn.Sigmoid()
        self.proj_norm = LayerNorm(opt["output_size"]*2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(opt["output_size"]*2,2)
        )

    def vit_forward(self,x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x[:,1:]

    # forward propagate input
    def forward(self, input):
        # (bs, max_len, dim)
        bert_embed_text = self.bertl_text.embeddings(input_ids = input[self.input1])
        # (bs, max_len, dim)
        # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        for i in range(self.opt["roberta_layer"]):
            bert_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
            bert_embed_text = bert_text
        # (bs, grid_num, dim)
        img_feat = self.vit_forward(input[self.input2])

        out1, lang_emb, img_emb = self.trar_text(img_feat, bert_embed_text,input[self.input3].unsqueeze(1).unsqueeze(2))
        out2, lang_emb, img_emb = self.trar_img(img_feat, bert_embed_text,input[self.input3].unsqueeze(1).unsqueeze(2))
        proj_feat = torch.concat((out1, out2), 1)
        # proj_feat = self.proj_norm(proj_feat)
        out = self.classifier(proj_feat)
        result = self.sigm(out)

        del bert_embed_text, bert_text, img_feat, out1, out, proj_feat, out2
    
        return result


def build_TRAR_Twostream_concat_multiTRAR_SA(opt,requirements):
    from transformers import RobertaModel
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    if "vitmodel" not in opt:
        opt["vitmodel"]="vit_base_patch32_224"
    vit = timm.create_model(opt["vitmodel"], pretrained=True)
    return TRAR_Twostream_concat(bertl_text,vit,opt)
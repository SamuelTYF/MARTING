import torch
import math
from model.TRAR import TRAR
import timm

class Co_attention(torch.nn.Module):
    def __init__(self, len):
        super().__init__()
        weights = torch.Tensor(768, 768)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.tanh = torch.nn.Tanh()
        self.max_pool = torch.nn.MaxPool2d(kernel_size = (len,1))

    def forward(self, H, T):
        batch=H.shape[0]
        T_tmp = T.clone()
        T = T.permute(0, 2, 1)
        z_1 = torch.matmul(H, self.weights)
        z_2 = torch.matmul(z_1, T)
        C = self.tanh(z_2)
        alpha = self.max_pool(C)
        HT = torch.matmul(alpha, T_tmp).reshape(batch, 768)
        return HT

class Roberta_CoAttention_TRAR(torch.nn.Module):
  # define model elements
    def __init__(self,bertl_text,bertl_attribute,vit, opt):
        super(Roberta_CoAttention_TRAR, self).__init__()

        self.bertl_text = bertl_text
        self.bertl_attribute = bertl_attribute
        self.vit=vit
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        assert("input4" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]
        self.input4=opt["input4"]

        self.trar = TRAR(opt)
        self.co_attention = Co_attention(opt["len"])
        self.sigm = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(768+opt["output_size"]+768,2)
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

        bert_embed_text = self.bertl_text.embeddings(input_ids = input[self.input1])

        bert_embed_attribute = self.bertl_attribute.embeddings(input_ids = input[self.input2])

        bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]

        img_feat = self.vit_forward(input[self.input3])

        out1, lang_emb, img_emb = self.trar(img_feat, bert_embed_text,input[self.input4].unsqueeze(1).unsqueeze(2))
        
        for i in range(12):
            bert_attribute = self.bertl_attribute.encoder.layer[i](bert_embed_attribute)[0]
            bert_embed_attribute = bert_attribute

        out2 = self.co_attention(bert_text, bert_attribute)

        out3 = bert_text[:,0,:]

        out = torch.cat((out1, out2, out3), dim = 1)
        out = self.classifier(out)
        result = self.sigm(out)

        del bert_embed_text, bert_embed_attribute, bert_text, img_feat, out1, out2, out3, out
    
        return result


def build_Roberta_CoAttention_TRAR(opt,requirements):
    from transformers import RobertaModel
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    bertl_attribute = RobertaModel.from_pretrained(opt["roberta_path"])
    if "vitmodel" not in opt:
        opt["vitmodel"]="vit_base_patch32_224"
    vit = timm.create_model(opt["vitmodel"], pretrained=True)
    return Roberta_CoAttention_TRAR(bertl_text,bertl_attribute,vit,opt)
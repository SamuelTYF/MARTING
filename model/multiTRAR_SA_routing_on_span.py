import torch
import timm
import model

class multiTRAR_SA_routing_on_span(torch.nn.Module):
  # define model elements
    def __init__(self,bertl_text,vit, opt):
        super(multiTRAR_SA_routing_on_span, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.vit=vit
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        self.trar = model.TRAR.multiTRAR_SA_routing_on_span_model(opt)
        self.sigm = torch.nn.Sigmoid()
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(opt["output_size"],2)
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
        # bert_embed_text = self.bertl_text.embeddings(input_ids = input[self.input1])
        # (bs, max_len, dim)
        # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        # for i in range(self.opt["roberta_layer"]):
        #     bert_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
        #     bert_embed_text = bert_text
        # del bert_text
        bert_encoding = self.bertl_text(input[self.input1]) 
        bert_embed_text = bert_encoding['last_hidden_state']
        # (bs, grid_num, dim)
        img_feat = self.vit_forward(input[self.input2])

        (out1, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text,input[self.input3].unsqueeze(1).unsqueeze(2))

        out = self.classifier(out1)
        result = self.sigm(out)

        del bert_embed_text, img_feat, out1, out
    
        return result, lang_emb, img_emb


def build_multiTRAR_SA_routing_on_span(opt,requirements):
    from transformers import RobertaModel
    bertl_text = RobertaModel.from_pretrained('roberta-base')
    if "vitmodel" not in opt:
        opt["vitmodel"]="vit_base_patch32_224"
    vit = timm.create_model(opt["vitmodel"], pretrained=True)
    return multiTRAR_SA_routing_on_span(bertl_text,vit,opt)

def build_multiTRAR_SA_routing_on_span_bert(opt,requirements):
    from transformers import BertModel
    bertl_text = BertModel.from_pretrained('bert-base-uncased')
    if "vitmodel" not in opt:
        opt["vitmodel"]="vit_base_patch32_224"
    vit = timm.create_model(opt["vitmodel"], pretrained=True)
    return multiTRAR_SA_routing_on_span(bertl_text,vit,opt)
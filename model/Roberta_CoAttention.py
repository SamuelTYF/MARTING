import torch
import math

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

class Roberta_CoAttention(torch.nn.Module):
    def __init__(self,bertl_text,bertl_attribute, opt):
        super(Roberta_CoAttention, self).__init__()

        self.bertl_text = bertl_text
        self.bertl_attribute = bertl_attribute
        assert("input1" in opt)
        assert("input2" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]

        if "len" not in opt:
            opt["len"]=360
        self.co_attention = Co_attention(opt["len"])
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(768+768,2)
        )
        self.sigm = torch.nn.Sigmoid()

    def forward(self, input):

        bert_embed_text = self.bertl_text.embeddings(input_ids = input[self.input1])

        bert_embed_attribute = self.bertl_attribute.embeddings(input_ids = input[self.input2])

        bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        
        for i in range(12):
            bert_attribute = self.bertl_attribute.encoder.layer[i](bert_embed_attribute)[0]
            bert_embed_attribute = bert_attribute

        out2 = self.co_attention(bert_text, bert_attribute)

        out3 = bert_text[:,0,:]

        out = torch.cat((out2, out3), dim = 1)
        out = self.classifier(out)
        result = self.sigm(out)

        del bert_embed_attribute,bert_embed_text,bert_text,out2,out3,out
        
        return result

def build_Roberta_CoAttention(opt,requirements):
    from transformers import RobertaModel
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    bertl_attribute = RobertaModel.from_pretrained(opt["roberta_path"])
    return Roberta_CoAttention(bertl_text,bertl_attribute,opt)
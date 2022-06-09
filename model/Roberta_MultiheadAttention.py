import torch
import math

class MHAtt(torch.nn.Module):
    def __init__(self, hidden_size, multihead, dropout):
        super(MHAtt, self).__init__()

        self.multihead=multihead
        self.hidden_size=hidden_size

        self.linear_v = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_k = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_q = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_merge = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(dropout)
        
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, v, k, q):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.multihead,
            int(self.hidden_size / self.multihead)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.multihead,
            int(self.hidden_size / self.multihead)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.multihead,
            int(self.hidden_size / self.multihead)
        ).transpose(1, 2)

        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        att_map = self.softmax(scores)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class Roberta_MultiheadAttention(torch.nn.Module):
    def __init__(self,bertl_text,bertl_attribute, opt):
        super(Roberta_MultiheadAttention, self).__init__()

        self.bertl_text = bertl_text
        self.bertl_attribute = bertl_attribute
        assert("input1" in opt)
        assert("input2" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]

        if "len" not in opt:
            opt["len"]=360
        if "num_heads" not in opt:
            opt["num_heads"]=4
        self.attention = MHAtt(768,opt["num_heads"],dropout=0.5)
        self.pool=torch.nn.MaxPool2d((opt["len"],1))
        
        self.classifier = torch.nn.Sequential(
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

        out2 = self.attention(bert_attribute,bert_attribute,bert_text)

        out2 = self.pool(out2).view(-1,768)

        out3 = bert_text[:,0,:]

        out = torch.cat((out2, out3), dim = 1)
        out = self.classifier(out)
        result = self.sigm(out)

        del bert_embed_attribute,bert_embed_text,bert_text,out2,out3,out
        
        return result

def build_Roberta_MultiheadAttention(opt,requirements):
    from transformers import RobertaModel
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    bertl_attribute = RobertaModel.from_pretrained(opt["roberta_path"])
    return Roberta_MultiheadAttention(bertl_text,bertl_attribute,opt)
import torch

class Roberta_MultiheadAttentionVariant(torch.nn.Module):
    def __init__(self,bertl_text,bertl_attribute, opt):
        super(Roberta_MultiheadAttentionVariant, self).__init__()

        self.bertl_text = bertl_text
        self.bertl_attribute = bertl_attribute
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        if "len" not in opt:
            opt["len"]=360
        if "num_heads" not in opt:
            opt["num_heads"]=4
        self.attention = torch.nn.MultiheadAttention(768,opt["num_heads"],batch_first=True,dropout=0)
        self.pool=torch.nn.AvgPool2d((opt["len"],1))
        
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

        _,out2 = self.attention(bert_text, bert_attribute*(input[self.input3].unsqueeze(2).expand_as(bert_attribute)),bert_attribute)

        out2 = self.pool(out2)*input[self.input3].unsqueeze(1)

        out2 = torch.matmul(out2,bert_attribute).squeeze(1)

        out3 = bert_text[:,0,:]

        out = torch.cat((out2, out3), dim = 1)
        out = self.classifier(out)
        result = self.sigm(out)

        del bert_embed_attribute,bert_embed_text,bert_text,out2,out3,out
        
        return result

def build_Roberta_MultiheadAttentionVariant(opt,requirements):
    from transformers import RobertaModel
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    bertl_attribute = RobertaModel.from_pretrained(opt["roberta_path"])
    return Roberta_MultiheadAttentionVariant(bertl_text,bertl_attribute,opt)
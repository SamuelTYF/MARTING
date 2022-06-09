from model.FiLM import FiLMGen, FiLMedNet
import torch
import math
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

class Roberta_CoAttention_FiLM(torch.nn.Module):
    def __init__(self,bertl_text,bertl_attribute,resnet,gen,film, opt):
        super(Roberta_CoAttention_FiLM, self).__init__()

        self.bertl_text = bertl_text
        self.bertl_attribute = bertl_attribute
        self.resnet=resnet
        self.gen=gen
        self.film=film
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        if "len" not in opt:
            opt["len"]=360
        self.co_attention = Co_attention(opt["len"])
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(768+768+1024,2)
        )
        self.sigm = torch.nn.Sigmoid()

    def forward(self, input):

        bert_embed_text = self.bertl_text.embeddings(input_ids = input[self.input1])

        bert_embed_attribute = self.bertl_attribute.embeddings(input_ids = input[self.input2])

        bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        
        bert_attribute = self.bertl_attribute.encoder.layer[0](bert_embed_attribute)[0]

        out2 = self.co_attention(bert_text, bert_attribute)

        out3 = bert_text[:,0,:]

        img = self.resnet(input[self.input3])

        gammas,betas=self.gen(out3)

        out4 = self.film(img,gammas,betas)

        out = torch.cat((out2, out3,out4), dim = 1)
        out = self.classifier(out)
        result = self.sigm(out)

        del bert_embed_attribute,bert_embed_text,bert_text,out2,out3,out4,out
        
        return result

def build_Roberta_CoAttention_FiLM(opt,requirements):
    from transformers import RobertaModel
    from torchvision import models
    bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    bertl_attribute = RobertaModel.from_pretrained(opt["roberta_path"])
    
    if "num_models" not in opt:
        opt["num_models"]=4

    if "model_dim" not in opt:
        opt["model_dim"]=128

    gen=FiLMGen(768,opt["num_models"],opt["model_dim"])

    if "dropout" not in opt:
        opt["dropout"]=0

    if "downsample" not in opt:
        opt["downsample"]="maxpool"

    film=FiLMedNet(opt["num_models"],opt["model_dim"],opt["dropout"],512,opt["downsample"],1024,opt["dropout"])

    cnn = models.resnet50(pretrained = True)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(3):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))

    resnet=torch.nn.Sequential(*layers)

    return Roberta_CoAttention_FiLM(bertl_text,bertl_attribute,resnet,gen,film,opt)
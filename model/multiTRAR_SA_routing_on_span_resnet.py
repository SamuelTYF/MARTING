import torch
import model
import torchvision

class multiTRAR_SA_routing_on_span_resnet(torch.nn.Module):
  # define model elements
    def __init__(self,bertl_text,resnet, opt):
        super(multiTRAR_SA_routing_on_span_resnet, self).__init__()

        self.bertl_text = bertl_text
        self.opt = opt
        self.resnet=resnet
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]

        self.trar = model.TRAR.multiTRAR_SA_routing_on_span_model(opt)
        self.sigm = torch.nn.Sigmoid()
        self.fc=torch.nn.Linear(2048,768)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(opt["output_size"],2)
        )

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
        img_feat =self.fc(self.resnet(input[self.input2]).view(-1,2048,49).permute(0,2,1))

        (out1, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text,input[self.input3].unsqueeze(1).unsqueeze(2))

        out = self.classifier(out1)
        result = self.sigm(out)

        del bert_embed_text, img_feat, out1, out
    
        return result, lang_emb, img_emb


def build_multiTRAR_SA_routing_on_span_resnet(opt,requirements):
    from transformers import RobertaModel
    bertl_text = RobertaModel.from_pretrained('roberta-base')
    cnn = torchvision.models.resnet50(pretrained = True)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(4):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    resnet=torch.nn.Sequential(*layers)
    return multiTRAR_SA_routing_on_span_resnet(bertl_text,resnet,opt)
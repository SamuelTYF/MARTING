import pickle
import re
import torch

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_vit:
    def __init__(self):
        self.name="vit"
        self.require=[]

    def prepare(self,input,opt):
        self.filenames ={
            "train":load_file(opt["data_path"] + "train_id"),
            "test":load_file(opt["data_path"] + "test_id"),
            "valid":load_file(opt["data_path"] + "valid_id")
        }
        if "model" not in opt:
            opt["model"]="vit_base_patch32_224"
        self.attributes=load_file(opt["data_path"] + ""+opt["model"]+"_attribute")
        self.probabilities=load_file(opt["data_path"] + ""+opt["model"]+"_probs")
        self.features=load_file(opt["data_path"] + ""+opt["model"]+"_features")
        if "len" not in opt:
            opt["len"]=13
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        if "weight_pad" not in opt:
            opt["weight_pad"]=0.1
        self.weight_pad=opt["weight_pad"]
        if "tokenizer" not in opt:
            opt["tokenizer"]="tokenizer_roberta"
        self.tokenizer_roberta=input[opt["tokenizer"]]

    def get(self,result,mode,idx):
        attribute = self.attributes[self.filenames[mode][idx]]
        indexed_tokens_for_attribute = self.tokenizer_roberta(' '.join(attribute))['input_ids']
        if len(indexed_tokens_for_attribute) > self.len:
            indexed_tokens_for_attribute=indexed_tokens_for_attribute[0:self.len]
        indexed_tokens_for_attribute+=[self.pad]*(self.len-len(indexed_tokens_for_attribute))
        weight=[1]
        probability=self.probabilities[self.filenames[mode][idx]]
        for i in range(len(attribute)):
            ids = self.tokenizer_roberta(attribute[i])['input_ids']
            weight+=[probability[i]/(len(ids)-2)]*(len(ids)-2)
        if len(weight) > self.len:
            weight=weight[0:self.len]
        for i in range(len(weight)):
            weight[i]/=len(weight)
        weight+=[self.weight_pad]*(self.len-len(weight))
        for i in range(self.len):
            weight[i]*=13
        result["vit_attribute"]=torch.tensor(indexed_tokens_for_attribute)
        result["vit_probability"]=torch.tensor(self.probabilities[self.filenames[mode][idx]])
        result["vit_feature"]=torch.tensor(self.features[self.filenames[mode][idx]])
        result["vit_weight"]=torch.FloatTensor(weight)

    def getlength(self,mode):
        return len(self.filenames[mode])
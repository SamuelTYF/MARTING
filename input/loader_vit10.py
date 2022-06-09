import pickle
import re
import torch

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_vit10:
    def __init__(self):
        self.name="vit10"
        self.require=[]

    def prepare(self,input,opt):
        self.filenames ={
            "train":load_file(opt["data_path"] + "train_id"),
            "test":load_file(opt["data_path"] + "test_id"),
            "valid":load_file(opt["data_path"] + "valid_id")
        }
        self.vit10=load_file(opt["data_path"] + "vit10")
        if "len" not in opt:
            opt["len"]=20
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        if "tokenizer" not in opt:
            opt["tokenizer"]="tokenizer_roberta"
        self.tokenizer_roberta=input[opt["tokenizer"]]

    def get(self,result,mode,idx):
        attribute = self.vit10[self.filenames[mode][idx]]
        attribute = ' '.join(attribute)
        indexed_tokens_for_attribute = self.tokenizer_roberta(attribute)['input_ids']
        if len(indexed_tokens_for_attribute) > self.len:
            indexed_tokens_for_attribute=indexed_tokens_for_attribute[0:self.len]
        indexed_tokens_for_attribute+=[self.pad]*(self.len-len(indexed_tokens_for_attribute))
        result["vit10"]=torch.tensor(indexed_tokens_for_attribute)

    def getlength(self,mode):
        return len(self.filenames[mode])
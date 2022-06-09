import pickle
import re
import torch

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_attribute:
    def __init__(self):
        self.name="attribute"
        self.require=[]

    def prepare(self,input,opt):
        self.filenames ={
            "train":load_file(opt["data_path"] + "train_id"),
            "test":load_file(opt["data_path"] + "test_id"),
            "valid":load_file(opt["data_path"] + "valid_id")
        }
        self.image_top=load_file(opt["data_path"] + "image_top")
        if "len" not in opt:
            opt["len"]=13
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        if "tokenizer" not in opt:
            opt["tokenizer"]="tokenizer_roberta"
        self.tokenizer_roberta=input[opt["tokenizer"]]

    def get(self,result,mode,idx):
        attribute = self.image_top[int(self.filenames[mode][idx])]
        attribute = ' '.join(attribute)
        indexed_tokens_for_attribute = self.tokenizer_roberta(attribute)['input_ids']
        if len(indexed_tokens_for_attribute) > self.len:
            indexed_tokens_for_attribute=indexed_tokens_for_attribute[0:self.len]
        indexed_tokens_for_attribute+=[self.pad]*(self.len-len(indexed_tokens_for_attribute))
        result["attribute"]=torch.tensor(indexed_tokens_for_attribute)

    def getlength(self,mode):
        return len(self.filenames[mode])
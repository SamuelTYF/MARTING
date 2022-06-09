import pickle
import re
import torch

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_text:
    def __init__(self):
        self.name="text"
        self.require=[]

    def prepare(self,input,opt):
        self.text ={
            "train":load_file(opt["data_path"] + "train_text"),
            "test":load_file(opt["data_path"] + "test_text"),
            "valid":load_file(opt["data_path"] + "valid_text")
        }
        if "len" not in opt:
            opt["len"]=360
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        if "tokenizer" not in opt:
            opt["tokenizer"]="tokenizer_roberta"
        self.tokenizer_roberta=input[opt["tokenizer"]]

    def get(self,result,mode,index):
        indexed_tokens_for_text = self.tokenizer_roberta(re.sub(r'(\s)emoji\w+', '', self.text[mode][index]))['input_ids']
        if len(indexed_tokens_for_text) > self.len:
            indexed_tokens_for_text=indexed_tokens_for_text[0:self.len]
        result["text_mask"]=torch.BoolTensor([0]*len(indexed_tokens_for_text)+[1]*(self.len-len(indexed_tokens_for_text)))
        indexed_tokens_for_text+=[self.pad]*(self.len-len(indexed_tokens_for_text))
        result["text"]=torch.tensor(indexed_tokens_for_text)

    def getlength(self,mode):
        return len(self.text[mode])
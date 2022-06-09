import pickle
import re
import torch
import json

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

def load_json(filename):
    with open(filename, 'r') as filehandle:
        ret = json.load(filehandle)
        return ret

class loader_emojitext:
    def __init__(self):
        self.name="emojitxt"
        self.require=["tokenizer_roberta"]

    def prepare(self,input,opt):
        self.text ={
            "train":load_file(opt["data_path"] + "train_text"),
            "test":load_file(opt["data_path"] + "test_text"),
            "valid":load_file(opt["data_path"] + "valid_text")
        }
        self.emoji=load_json(opt["data_path"] + "emojimapped.json")
        if "len" not in opt:
            opt["len"]=360
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        self.tokenizer_roberta=input["tokenizer_roberta"]

    def replace(self,text):
        text=text.group(2)
        if text in self.emoji:
            return self.emoji[text]
        else:
            return "emoji"

    def get(self,result,mode,index):
        indexed_tokens_for_text = self.tokenizer_roberta(re.sub(r'(\s)(emoji\w+)', self.replace, self.text[mode][index]))['input_ids']
        if len(indexed_tokens_for_text) > self.len:
            indexed_tokens_for_text=indexed_tokens_for_text[0:self.len]
        indexed_tokens_for_text+=[self.pad]*(self.len-len(indexed_tokens_for_text))
        result["emojitxt"]=torch.tensor(indexed_tokens_for_text)

    def getlength(self,mode):
        return len(self.text[mode])
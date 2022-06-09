import pickle
import re
import torch

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_ocr_text_pair:
    def __init__(self):
        self.name="ocr_text_pair"
        self.require=["tokenizer_bert"]

    def prepare(self,input,opt):
        self.text ={
            "train":load_file(opt["data_path"] + "train_text"),
            "test":load_file(opt["data_path"] + "test_text"),
            "valid":load_file(opt["data_path"] + "valid_text")
        }
        self.id ={
            "train":load_file(opt["data_path"] + "train_id"),
            "test":load_file(opt["data_path"] + "test_id"),
            "valid":load_file(opt["data_path"] + "valid_id")
        }
        self.ocr_text=load_file(opt["data_path"] + "ocr_text")
        if "len" not in opt:
            opt["len"]=360
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        self.tokenizer_bert=input["tokenizer_bert"]

    def get(self,result,mode,index):
        r = self.tokenizer_bert(self.text[mode][index],self.ocr_text[self.id[mode][index]])
        indexed_tokens_for_text=r['input_ids']
        token_type_ids=r['input_ids']
        if len(indexed_tokens_for_text)>self.len:
            indexed_tokens_for_text=indexed_tokens_for_text[0:self.len]
            token_type_ids=token_type_ids[0:self.len]
        indexed_tokens_for_text+=[self.pad]*(self.len-len(indexed_tokens_for_text))
        token_type_ids+=[1]*(self.len-len(token_type_ids))
        result["ocr_text_pair"]=torch.tensor(indexed_tokens_for_text)
        result["ocr_text_pair_ids"]=torch.tensor(token_type_ids)
        if len(indexed_tokens_for_text)!=self.len or len(token_type_ids)!=self.len:
            print(len(indexed_tokens_for_text),len(token_type_ids))

    def getlength(self,mode):
        return len(self.text[mode])
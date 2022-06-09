import pickle
import re
import torch

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_ocr_text:
    def __init__(self):
        self.name="ocr_text"
        self.require=["tokenizer_roberta"]

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
        self.ocr_text=load_file(opt["data_path"] + "ocr_para_clean")
        if "len" not in opt:
            opt["len"]=360
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        self.tokenizer_roberta=input["tokenizer_roberta"]

    def get(self,result,mode,index):
        r = self.tokenizer_roberta(re.sub(r'(\s)emoji\w+', '', self.text[mode][index]) +  self.ocr_text[self.id[mode][index]])
        indexed_tokens_for_text=r['input_ids']
        result["text_mask"]=torch.BoolTensor([0]*len(indexed_tokens_for_text)+[1]*(self.len-len(indexed_tokens_for_text)))
        indexed_tokens_for_text+=[self.pad]*(self.len-len(indexed_tokens_for_text))
        result["ocr_text"]=torch.tensor(indexed_tokens_for_text)

    def getlength(self,mode):
        return len(self.text[mode])
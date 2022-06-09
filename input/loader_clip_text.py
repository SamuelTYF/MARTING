import pickle
import clip
import torch
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_clip_text:
    def __init__(self):
        self.name="clip_text"
        self.require=[]
        self._tokenizer=_Tokenizer()

    def prepare(self,input,opt):
        self.text ={
            "train":load_file(opt["data_path"] + "train_text"),
            "test":load_file(opt["data_path"] + "test_text"),
            "valid":load_file(opt["data_path"] + "valid_text")
        }

    def get(self,result,mode,index):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = self._tokenizer.encode(self.text[mode][index])
        if len(all_tokens)>75:
            all_tokens=all_tokens[:75]
        result["clip_text"]=torch.LongTensor([sot_token]+all_tokens+[eot_token]+[0]*(75-len(all_tokens)))
    def getlength(self,mode):
        return len(self.text[mode])
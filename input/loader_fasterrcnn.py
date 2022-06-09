import pickle
import re
import torch
import os
from PIL import Image
from torchvision import transforms

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_fasterrcnn:
    def __init__(self):
        self.name="fasterrcnn"
        self.require=[]

    def prepare(self,input,opt):
        self.id ={
            "train":load_file(opt["data_path"] + "train_id"),
            "test":load_file(opt["data_path"] + "test_id"),
            "valid":load_file(opt["data_path"] + "valid_id")
        }
        self.boxfeature_path=opt["boxfeature_path"] 

    def get(self,result,mode,index):
        boxfeature_path=os.path.join(
                self.boxfeature_path,
                "{}".format(self.id[mode][index])
            )
        result["fasterrcnn"]=load_file(boxfeature_path)

    def getlength(self,mode):
        return len(self.id[mode])
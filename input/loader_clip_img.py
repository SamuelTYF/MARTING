import pickle
import re
import torch
import os
from PIL import Image
def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_clip_img:
    def __init__(self):
        self.name="clip_img"
        self.require=["clip_preprocess"]

    def prepare(self,input,opt):
        self.id ={
            "train":load_file(opt["data_path"] + "train_id"),
            "test":load_file(opt["data_path"] + "test_id"),
            "valid":load_file(opt["data_path"] + "valid_id")
        }
        self.img_dir=opt["img_path"] 
        self.clip_preprocess=input["clip_preprocess"]

    def get(self,result,mode,index):
        img_path=os.path.join(
                self.img_dir,
                "{}.jpg".format(self.id[mode][index])
            )
        img = Image.open(img_path)
        img = img.convert('RGB') # convert grey picture
        result["clip_img"]=self.clip_preprocess(img)

    def getlength(self,mode):
        return len(self.id[mode])
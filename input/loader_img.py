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

class loader_img:
    def __init__(self):
        self.name="img"
        self.require=[]

    def prepare(self,input,opt):
        self.id ={
            "train":load_file(opt["data_path"] + "train_id"),
            "test":load_file(opt["data_path"] + "test_id"),
            "valid":load_file(opt["data_path"] + "valid_id")
        }
        self.img_dir=opt["img_path"] 

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_valid = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.transform = {
            "train": transform_train,
            "valid": transform_valid,
            "test": transform_test
        }

    def get(self,result,mode,index):
        img_path=os.path.join(
                self.img_dir,
                "{}.jpg".format(self.id[mode][index])
            )
        img = Image.open(img_path)
        img = img.convert('RGB') # convert grey picture

        result["img"]=self.transform[mode](img)

    def getlength(self,mode):
        return len(self.id[mode])
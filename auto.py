import os
import sys

def main(args):
    print(args)
    assert(len(args)==2)
    fname="./list/"+args[1]
    assert(os.path.exists(fname) and fname.endswith(".txt"))
    with open(fname,"r") as f:
        for config in f.readlines():
            os.system("python train.py "+config)

if __name__=="__main__":
    main(sys.argv)
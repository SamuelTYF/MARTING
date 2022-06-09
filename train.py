from modelviewer import client
import os
import sys

def main(args):
    print(args)
    if len(args)==2:
        fname="./options/"+args[1]
        assert(os.path.exists(fname) and fname.endswith(".json"))
    elif len(args)==1:
        list=sorted(os.listdir("./options"))
        for i in range(len(list)):
            print("[%d]\t%s"%(i,list[i]))
        index=int(input("Select One Json:"))
        assert((0<=index) and (index<len(list)))
        fname="./options/"+list[index]
    else:
        raise NotImplemented()
    host=client(fname)
    host.start()

if __name__=="__main__":
    main(sys.argv)
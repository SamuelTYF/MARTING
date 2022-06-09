import os
import argparse
import datetime
import yaml
import json
import glob

from modelviewer import client
import sys


def main(args):
    print(args)
    assert(len(args)==2)
    fname=args[1]
    # fname = "config-daily/22-04/multiTRAR_SA_text_guide_order0123_batch32_TRAT_layer6_trarlr1e-4.json"
    assert(os.path.exists(fname) and fname.endswith(".json"))
    host=client(fname)
    host.start()

    

if __name__=="__main__":
    main(sys.argv)


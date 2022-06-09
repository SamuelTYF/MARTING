import torch


def get_tokenizer_roberta(results,opt):
    from transformers import RobertaTokenizer
    results["tokenizer_roberta"]=RobertaTokenizer.from_pretrained('roberta-base')

def get_tokenizer_bert(results,opt):
    from transformers import BertTokenizer
    results["tokenizer_bert"]=BertTokenizer.from_pretrained('bert-base-uncased')

def get_clip(results,opt):
    assert("modelname" in opt)
    assert("device" in opt)
    import clip
    results["clip"],results["clip_preprocess"]= clip.load(opt["modelname"],device=torch.device(opt["device"]))
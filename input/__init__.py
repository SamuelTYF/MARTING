from input.loader_ocr_text import loader_ocr_text
from input.loader_text import loader_text
from input.loader_attribute import loader_attribute
from input.loader_label import loader_label
from input.loader_ocr_text import loader_ocr_text
from input.loader_ocr_text_pair import loader_ocr_text_pair
from input.loader_clip_img import loader_clip_img
from input.loader_clip_text import loader_clip_text
from input.requires import get_tokenizer_roberta,get_tokenizer_bert,get_clip
from input.loader_vit5 import loader_vit5
from input.loader_vit10 import loader_vit10
from input.loader_clip_ocrtext import loader_clip_ocrtext
from input.loader_emojitext import loader_emojitext
from input.loader_vit import loader_vit
from input.loader_img import loader_img
from input.loader_text_imagetop import loader_text_imagetop
from input.loader_fasterrcnn import loader_fasterrcnn
_loaders=[
    loader_text(),
    loader_attribute(),
    loader_label(),
    loader_ocr_text(),
    loader_ocr_text_pair(),
    loader_vit5(),
    loader_vit10(),
    loader_clip_img(),
    loader_clip_text(),
    loader_clip_ocrtext(),
    loader_emojitext(),
    loader_vit(),
    loader_img(),
    loader_text_imagetop(),
    loader_fasterrcnn()
]
_requires={
    "tokenizer_roberta":get_tokenizer_roberta,
    "tokenizer_bert":get_tokenizer_bert,
    "clip":get_clip
}

_loadermap={}

for loader in _loaders:
    _loadermap[loader.name]=loader


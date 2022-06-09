from xmlrpc.client import Binary
import torch
from model.Roberta_CoAttention import build_Roberta_CoAttention
from model.Bert_CoAttention import build_Bert_CoAttention
from model.Clip import build_Clip
from model.Optimizers import build_Adam,build_SGD
from model.ClipS import build_ClipS
from model.Roberta_CoAttention_Encoder import build_Roberta_CoAttention_Encoder
from model.Roberta_CoAttention_Vit import build_Roberta_CoAttention_Vit
from model.Roberta_CoAttention_FiLM import build_Roberta_CoAttention_FiLM
from model.Roberta_MultiheadAttention import build_Roberta_MultiheadAttention
from model.Roberta_CoAttentionVariant import build_Roberta_CoAttentionVariant
from model.Roberta_MultiheadAttentionVariant import build_Roberta_MultiheadAttentionVariant
from model.Roberta_Vit_CoAttention import build_Roberta_Vit_CoAttention
from model.Roberta_CoAttention_TRAR import build_Roberta_CoAttention_TRAR
from model.Simply_TRAR import build_Simply_TRAR
from model.TRAR_Twostream_add import build_TRAR_Twostream_add
from model.TRAR_Twostream_concat import build_TRAR_Twostream_concat
from model.multiTRAR_SA import build_multiTRAR_SA,build_multiTRAR_SA_bert
from model.SA_multiTRAR import build_SA_multiTRAR
from model.BiTRAR_crossSA import build_BiTRAR_crossSA
from model.Bi_multiTRAR_SA import build_Bi_multiTRAR_SA
from model.Encoder_TRAR_Decoder_multiTRAR_SA import build_Encoder_TRAR_Decoder_multiTRAR_SA
from model.loss_function import build_CrossentropyLoss_ContrastiveLoss, build_BCELoss, build_CrossEntropyLoss, build_CrossEntropyLoss_weighted
from model.multiTRAR_SA_concat_roberta0 import build_multiTRAR_SA_concat_roberta0
from model.multiTRAR_SA_routing_on_span import build_multiTRAR_SA_routing_on_span,build_multiTRAR_SA_routing_on_span_bert
from model.multiTRAR_SA_routing_on_span_fasterrcnn import build_multiTRAR_SA_routing_on_span_fasterrcnn
from model.multiTRAR_SA_routing_on_span_resnet import build_multiTRAR_SA_routing_on_span_resnet
from model.Dual_multiTRAR_SA import build_Dual_multiTRAR_SA
from model.multiTRAR_SA_fasterrcnn import build_multiTRAR_SA_fasterrcnn
_models={
    "Roberta_CoAttention":build_Roberta_CoAttention,
    "Bert_CoAttention":build_Bert_CoAttention,
    "Clip":build_Clip,
    "ClipS":build_ClipS,
    "Roberta_CoAttention_Encoder":build_Roberta_CoAttention_Encoder,
    "Roberta_CoAttention_Vit":build_Roberta_CoAttention_Vit,
    "Roberta_CoAttention_FiLM":build_Roberta_CoAttention_FiLM,
    "Roberta_MultiheadAttention":build_Roberta_MultiheadAttention,
    "Roberta_CoAttentionVariant":build_Roberta_CoAttentionVariant,
    "Roberta_MultiheadAttentionVariant":build_Roberta_MultiheadAttentionVariant,
    "Roberta_Vit_CoAttention":build_Roberta_Vit_CoAttention,
    "Roberta_CoAttention_TRAR":build_Roberta_CoAttention_TRAR,
    "Simply_TRAR":build_Simply_TRAR,
    "TRAR_Twostream_add":build_TRAR_Twostream_add,
    "TRAR_Twostream_concat":build_TRAR_Twostream_concat,
    "multiTRAR_SA":build_multiTRAR_SA,
    "SA_multiTRAR":build_SA_multiTRAR,
    "BiTRAR_crossSA":build_BiTRAR_crossSA,
    "Bi_multiTRAR_SA":build_Bi_multiTRAR_SA,
    "Encoder_TRAR_Decoder_multiTRAR_SA":build_Encoder_TRAR_Decoder_multiTRAR_SA,
    "multiTRAR_SA_concat_roberta0": build_multiTRAR_SA_concat_roberta0,
    "multiTRAR_SA_routing_on_span": build_multiTRAR_SA_routing_on_span,
    "multiTRAR_SA_routing_on_span_bert":build_multiTRAR_SA_routing_on_span_bert,
    "multiTRAR_SA_routing_on_span_fasterrcnn":build_multiTRAR_SA_routing_on_span_fasterrcnn,
    "multiTRAR_SA_routing_on_span_resnet":build_multiTRAR_SA_routing_on_span_resnet,
    "Dual_multiTRAR_SA":build_Dual_multiTRAR_SA,
    "multiTRAR_SA_bert":build_multiTRAR_SA_bert,
    "multiTRAR_SA_fasterrcnn":build_multiTRAR_SA_fasterrcnn
}

_optimizers={
    "Adam":build_Adam,
    "SGD":build_SGD
}

_loss={
    "CrossEntropyLoss":build_CrossEntropyLoss,
    "BCELoss":build_BCELoss,
    "CrossentropyLoss_ContrastiveLoss": build_CrossentropyLoss_ContrastiveLoss,
    "Crossentropy_Loss_weighted": build_CrossEntropyLoss_weighted
}
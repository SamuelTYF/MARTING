import torch
from model.TRAR.trar import TRAR_ED_img_guide, TRAR_ED_txt_guide, AttFlat, multiTRAR_SA_ED_text_guide, multiTRAR_SA_ED_img_guide, SA_multiTRAR_ED_img_guide,  SA_multiTRAR_ED_text_guide, BiTRAR_crossSA_ED, Bi_multiTRAR_SA_ED, multiTRAR_SA_TRARED_text_guide, multiTRAR_SA_TRARED_img_guide, multiTRAR_SA_routing_on_span_ED_text_guide, multiTRAR_SA_routing_on_span_ED_img_guide
from model.TRAR.cls_layer import cls_layer_both, cls_layer_img, cls_layer_txt, cls_layer_both_concat, cls_layer_both_routing
import torch.nn as nn

class TRAR(nn.Module):
    def __init__(self, opt):
        super(TRAR, self).__init__()

        if opt["backbone"] == 'text_guide':
            self.backbone = TRAR_ED_txt_guide(opt)
        else:
            self.backbone = TRAR_ED_img_guide(opt)

        self.attflat_img = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])
        self.attflat_lang = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])

        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_concat':
            self.cls_layer = cls_layer_both_concat(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_routing':
            self.cls_layer = cls_layer_both_routing(opt["hidden_size"],opt["output_size"], opt["pooling"])
        elif opt["classifier"] == 'lang':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat

class BiTRAR_crossSA_model(nn.Module):
    def __init__(self, opt):
        super(BiTRAR_crossSA_model, self).__init__()

        self.backbone = BiTRAR_crossSA_ED(opt)

        self.attflat_img = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])
        self.attflat_lang = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])

        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_concat':
            self.cls_layer = cls_layer_both_concat(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_routing':
            self.cls_layer = cls_layer_both_routing(opt["hidden_size"],opt["output_size"], opt["pooling"])
        elif opt["classifier"] == 'lang':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat


class multiTRAR_SA_model(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_model, self).__init__()

        if opt["backbone"] == 'text_guide':
            self.backbone = multiTRAR_SA_ED_text_guide(opt)
        else:
            self.backbone = multiTRAR_SA_ED_img_guide(opt)

        self.attflat_img = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])
        self.attflat_lang = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])

        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_concat':
            self.cls_layer = cls_layer_both_concat(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_routing':
            self.cls_layer = cls_layer_both_routing(opt["hidden_size"],opt["output_size"], opt["pooling"])
        elif opt["classifier"] == 'lang':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat

class multiTRAR_SA_routing_on_span_model(nn.Module):
    def __init__(self, opt):
        super(multiTRAR_SA_routing_on_span_model, self).__init__()

        if opt["backbone"] == 'text_guide':
            self.backbone = multiTRAR_SA_routing_on_span_ED_text_guide(opt)
        else:
            self.backbone = multiTRAR_SA_routing_on_span_ED_img_guide(opt)

        self.attflat_img = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])
        self.attflat_lang = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])

        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_concat':
            self.cls_layer = cls_layer_both_concat(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_routing':
            self.cls_layer = cls_layer_both_routing(opt["hidden_size"],opt["output_size"], opt["pooling"])
        elif opt["classifier"] == 'lang':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat



class Bi_multiTRAR_SA_model(nn.Module):
    def __init__(self, opt):
        super(Bi_multiTRAR_SA_model, self).__init__()

        self.backbone = Bi_multiTRAR_SA_ED(opt)

        self.attflat_img = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])
        self.attflat_lang = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])

        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_concat':
            self.cls_layer = cls_layer_both_concat(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_routing':
            self.cls_layer = cls_layer_both_routing(opt["hidden_size"],opt["output_size"], opt["pooling"])
        elif opt["classifier"] == 'lang':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat



class SA_multiTRAR_model(nn.Module):
    def __init__(self, opt):
        super(SA_multiTRAR_model, self).__init__()

        if opt["backbone"] == 'text_guide':
            self.backbone = SA_multiTRAR_ED_text_guide(opt)
        else:
            self.backbone = SA_multiTRAR_ED_img_guide(opt)

        self.attflat_img = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])
        self.attflat_lang = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])

        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_concat':
            self.cls_layer = cls_layer_both_concat(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_routing':
            self.cls_layer = cls_layer_both_routing(opt["hidden_size"],opt["output_size"], opt["pooling"])
        elif opt["classifier"] == 'lang':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat


class Encoder_TRAR_Decoder_multiTRAR_SA(nn.Module):
    def __init__(self, opt):
        super(Encoder_TRAR_Decoder_multiTRAR_SA, self).__init__()

        if opt["backbone"] == 'text_guide':
            self.backbone = multiTRAR_SA_TRARED_text_guide(opt)
        else:
            self.backbone = multiTRAR_SA_TRARED_img_guide(opt)

        self.attflat_img = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])
        self.attflat_lang = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])

        if opt["classifier"] == 'both':
            self.cls_layer = cls_layer_both(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_concat':
            self.cls_layer = cls_layer_both_concat(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'both_routing':
            self.cls_layer = cls_layer_both_routing(opt["hidden_size"],opt["output_size"], opt["pooling"])
        elif opt["classifier"] == 'lang':
            self.cls_layer = cls_layer_txt(opt["hidden_size"],opt["output_size"])
        elif opt["classifier"] == 'img':
            self.cls_layer = cls_layer_img(opt["hidden_size"],opt["output_size"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = self.cls_layer(lang_feat, img_feat)

        return proj_feat, lang_feat, img_feat


class TRAR_one_stream(nn.Module):
    def __init__(self, opt, opt_backbone):
        super(TRAR_one_stream, self).__init__()

        if opt_backbone == 'text_guide':
            self.backbone = TRAR_ED_txt_guide(opt)
        else:
            self.backbone = TRAR_ED_img_guide(opt)

        self.attflat_img = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])
        self.attflat_lang = AttFlat(opt["hidden_size"],opt["mlp_size"],opt["hidden_size"],opt["glimpses"],opt["dropout"])


    def forward(self, img_feat, lang_feat, lang_feat_mask):
        img_feat_mask = torch.zeros([img_feat.shape[0],1,1,img_feat.shape[1]],dtype=torch.bool,device=img_feat.device)
        # (bs, 1, 1, grid_num)
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        return lang_feat, img_feat


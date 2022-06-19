from utils import logprint
from torch import nn
import torch
from networks import get_cond_unet_model
from transofmer_models import get_bert_model


class CondModel(nn.Module):
    def __init__(self, up_depths, last_conv_depth, backbone='vgg', img2txt=0, **kwargs):
        super(CondModel, self).__init__()
        self.caption_model = get_bert_model(out_features=None, average_outputs=True)
        cond_features = self.caption_model.get_depth()
        self.image_model = get_cond_unet_model(decoder_depths=up_depths, last_conv_depth=last_conv_depth,
                                               backbone=backbone,
                                               cond_features=cond_features,
                                               **kwargs)
        self.out_down_factor = self.image_model.out_down_factor
        self.img2txt = img2txt

    def forward(self, input_image, *inputs):
        captions = []
        captions_words = []
        for ind in range(len(inputs)):
            if not ind % 2:
                captions_words.append(self.caption_model(inputs[ind], mask=inputs[ind + 1], average_outputs=False))
                captions.append(self.caption_model.average(captions_words[-1], mask=inputs[ind + 1]))

        outputs = self.image_model(input_image, *captions)
        if self.img2txt:
            backbone_features = self.image_model.backbone_features
            word_embed = captions_words
            img2txt_heatmaps = []

            # image2text heatmaps
            pre_cond_features = \
                [pre_cond_layer(feature) for feature, pre_cond_layer in zip(backbone_features[1:],
                                                                            self.image_model.pre_cond)]
            resized_pre_cond_features = [self.image_model.scale_conv(feature) for feature in pre_cond_features]
            img_embed = torch.cat([pre_cond.unsqueeze(0) for pre_cond in resized_pre_cond_features], dim=0
                                  ).reshape((len(resized_pre_cond_features), resized_pre_cond_features[0].shape[0],
                                             resized_pre_cond_features[0].shape[1],
                                             resized_pre_cond_features[0].shape[-1]**2)).permute((1,3,0,2))

            for i in range(len(captions)):
                img2txt_heatmaps.append(self.image_model.get_im2txt_heatmap(img_embed, word_embed[i]))
            return outputs, img2txt_heatmaps

        return outputs, None


class CondAlphaMapTM(nn.Module):
    def __init__(self, up_depths, backbone='vgg', img2txt=0, *args, **kwargs):
        super().__init__()
        if len(args):
            logprint(f'Ignoring {len(args)} positional arguments')
        if len(kwargs):
            logprint(f'Ignoring {kwargs.keys()} key arguments')

        last_conv_depth = 1
        self.model = CondModel(up_depths=up_depths, last_conv_depth=last_conv_depth, backbone=backbone,
                               img2txt=img2txt, **kwargs)
        self.downsample = lambda x: torch.nn.functional.interpolate(x, scale_factor=1 / self.model.out_down_factor)
        self.upsample = lambda x: torch.nn.functional.interpolate(x, scale_factor=self.model.out_down_factor)

    def forward(self, input_image: torch.Tensor, *inputs):

        assert not (len(inputs) % 2)
        gbs_heatmaps, img2txt_heatmaps = self.model(input_image, *inputs)
        gbs_heatmaps = tuple(hm.clamp(0., 1.) for hm in gbs_heatmaps)
        if not img2txt_heatmaps is None:
            img2txt_heatmaps = tuple(hm.clamp(0., 1.) for hm in img2txt_heatmaps)
        return gbs_heatmaps, img2txt_heatmaps

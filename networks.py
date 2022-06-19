# noinspection PyProtectedMember
from torchvision.models.resnet import conv1x1, conv3x3
import torch
from torch import nn
import torch.nn.functional as torch_func
import numpy as np
from vgg import vgg16_bn
import conditions
import timm_backbones


class InterpBilinear(object):
    def __init__(self, size=None, scale_factor=None):
        self.size = size
        self.scale_factor = scale_factor
        self.mode = 'bilinear'
        self.align_corners = True

    def __call__(self, x):
        return torch.nn.functional.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class CondUnetBase(nn.Module):
    def __init__(self, in_depth, decoder_depths, last_conv_depth, downsample=nn.AvgPool2d,
                 upsample=nn.UpsamplingBilinear2d, cond_features=None, input_size=512, **kwargs):
        super(CondUnetBase, self).__init__()
        self.down_blocks = None
        self.out_down_factor = 1
        self.output_size = input_size
        self.decoder_blocks = torch.nn.ModuleList([])
        self.conditions = torch.nn.ModuleList()
        self.no_conditions = torch.nn.ModuleList()
        self.decoder_depths = decoder_depths
        self.input_depth = in_depth
        self.input_size = input_size
        self.cond_features = cond_features
        self.upsample = upsample
        self.downsample = downsample
        self.last_conv_depth = last_conv_depth
        self.last_conv = None
        self.scale_conv = None
        self.backbone_features = None
        self.decoder_features = None
        self.scale_conv = None
        self.pre_cond = torch.nn.ModuleList([])
        self.pre_cond_up = torch.nn.ModuleList([])
        self.bottle_neck_out = 512

    def build(self):
        down_factor = np.prod(self.down_sample_list)
        if len(self.decoder_depths):
            up_factor = np.prod(self.down_sample_list[-len(self.decoder_depths):])
            self.out_down_factor = down_factor / up_factor
        else:
            self.out_down_factor = down_factor
        self.output_size = self.input_size / self.out_down_factor
        num_conds = len(self.decoder_depths) + 1

        for depth in self.down_depths:
            self.pre_cond.append(BottleneckUp(depth, self.bottle_neck_out, stride=1, upsample=self.upsample(size=32)))
        for depth in self.decoder_depths:
            self.pre_cond_up.append(BottleneckUp(depth, self.bottle_neck_out, stride=1,
                                                 upsample=self.upsample(size=32)))
        condition_depths = [self.input_depth] + self.down_depths
        for ind in range(num_conds):
            cond = conditions.DistCond(cap_in_depth=self.cond_features,
                                       out_depth=condition_depths[-1 - ind])
            self.conditions.append(cond)
            self.no_conditions.append(conditions.NoCond(cap_in_depth=self.cond_features,
                                                        out_depth=condition_depths[-1-ind]))
        in_depth = 0
        l_ind = -1
        for l_ind, (out_depth, bridge_depth) in enumerate(zip(self.decoder_depths, reversed(condition_depths))):
            if l_ind == 0:
                in_depth = 0

            self.decoder_blocks.append(BottleneckUp(in_depth + bridge_depth, out_depth, stride=1,
                                                    upsample=None))
            in_depth = out_depth
        bridge_depth = condition_depths[-(l_ind+2)]
        self.scale_conv = conv1x1(self.bottle_neck_out, self.cond_features)
        self.last_conv = conv3x3(in_depth + bridge_depth, self.last_conv_depth)

    def forward(self, *inputs, **kwargs):
        x = inputs[0]
        caption_embedding_conditions = torch.stack(inputs[1:], dim=0)
        num_outputs = len(caption_embedding_conditions)

        backbone_features = self.get_backbone_features(x)
        self.backbone_features = backbone_features
        decoder_features = [[] for _ in range(num_outputs)]
        outputs = []
        rev_backbone = list(reversed(backbone_features))[:len(self.decoder_blocks) + 1]

        conditioned_down = [cond(dout, caption_embedding_conditions)
                            for cond, dout in zip(self.conditions, rev_backbone)]

        for cond_num in range(len(decoder_features)):
            this_rev_backbone = rev_backbone.copy()
            for skip_ind, conds in enumerate(conditioned_down):
                this_rev_backbone[skip_ind] = conds[cond_num]

            for ind, (decoder_layer, skip) in enumerate(zip(list(self.decoder_blocks) + [self.last_conv],
                                                            this_rev_backbone)):
                if ind == 0:  # Deepest encoder level
                    this_input = skip
                else:  # All other encoder level
                    skip_shape = skip.shape[2:4]
                    last_shape = decoder_features[cond_num][-1].shape[2:4]
                    if not (skip_shape[0] == last_shape[0] and skip_shape[1] == last_shape[1]):
                        decoder_features[cond_num][-1] = torch.nn.functional.interpolate(decoder_features[cond_num][-1],
                                                                                         skip_shape, mode='bilinear')
                    this_input = torch.cat((skip, decoder_features[cond_num][-1]), dim=1)

                if decoder_layer is not None:
                    this_input = decoder_layer(this_input)
                decoder_features[cond_num].append(this_input)
            self.decoder_features = decoder_features
            outputs.append(decoder_features[cond_num][-1])

        run_no_cond = kwargs.get('run_no_cond', False)
        if run_no_cond:
            no_conditioned_down = [cond(dout, None) for cond, dout in zip(self.no_conditions, rev_backbone)]
            this_rev_backbone = rev_backbone.copy()
            for skip_ind, conds in enumerate(no_conditioned_down):
                this_rev_backbone[skip_ind] = conds
            decoder_features.append([])
            for ind, (decoder_layer, skip) in enumerate(zip(list(self.decoder_blocks) + [self.last_conv],
                                                            this_rev_backbone)):
                if ind == 0:
                    this_input = skip
                else:
                    skip_shape = skip.shape[2:4]
                    last_shape = decoder_features[-1][-1].shape[2:4]
                    if not (skip_shape[0] == last_shape[0] and skip_shape[1] == last_shape[1]):
                        decoder_features[-1][-1] = torch.nn.functional.interpolate(decoder_features[-1][-1], skip_shape,
                                                                                   mode='bilinear')
                    this_input = torch.cat((skip, decoder_features[-1][-1]), dim=1)

                decoder_features[-1].append(decoder_layer(this_input))
            outputs.append(decoder_features[-1][-1])

        return tuple(outputs)

    def get_backbone_features(self, x, *args):
        self.backbone(x)
        return self.backbone.backbone_features

    @staticmethod
    def get_im2txt_heatmap(image_embedding, word_embedding):
        word_embedding = torch_func.normalize(word_embedding, dim=-1)
        image_embedding = torch_func.normalize(image_embedding, dim=-1)

        heatmaps = torch_func.relu(torch.matmul(word_embedding.unsqueeze(1), image_embedding.permute((0, 1, 3, 2))).permute(0, 2, 1, 3))
        attention = torch.einsum('bijk,bjkl->bilk', heatmaps, image_embedding)
        attention = torch_func.normalize(attention, dim=2)
        attention_word_sim = torch.matmul(attention.permute(0,1,3,2), word_embedding.unsqueeze(-1)).squeeze(-1)
        attention_word_sim_max = torch.max(attention_word_sim, dim=-1)[0]
        attention_word_sim_argmax = torch.argmax(attention_word_sim, dim=-1)
        sim_i, sim_j = torch.meshgrid(torch.arange(attention_word_sim_argmax.shape[0], dtype=torch.int64).cuda(),
                                      torch.arange(attention_word_sim_argmax.shape[1], dtype=torch.int64).cuda())
        heatmaps_max = heatmaps.permute(0, 1, 3, 2)[sim_i.reshape(-1), sim_j.reshape(-1), attention_word_sim_argmax.reshape(-1), :]
        heatmaps_inner_dim = int(np.sqrt(heatmaps.shape[2]))
        word_heatmaps = heatmaps_max.reshape(heatmaps.shape[0], heatmaps.shape[1], heatmaps_inner_dim, heatmaps_inner_dim)

        normalized_scores = attention_word_sim_max / (torch.sum(attention_word_sim_max, dim=1).view(attention_word_sim_max.shape[0], 1) + 1e-6)
        final_heatmap = torch.sum(word_heatmaps * normalized_scores.view(normalized_scores.shape[0],
                                                                         normalized_scores.shape[1], 1, 1),
                                  dim=1).unsqueeze(1)
        return final_heatmap


class CondUNetVGG16(CondUnetBase):
    def __init__(self, in_depth, decoder_depths, last_conv_depth, downsample=nn.AvgPool2d,
                 upsample=nn.UpsamplingBilinear2d, cond_features=None, input_size=512,
                 pretrained=True, **kwargs):
        super(CondUNetVGG16, self).__init__(in_depth, decoder_depths, last_conv_depth, downsample, upsample,
                                            cond_features, input_size, **kwargs)
        self.backbone = vgg16_bn(pretrained=pretrained).features
        self.down_depths = [64, 128, 256, 512, 512]
        self.down_sample_list = np.array([2, 2, 2, 2, 2])
        self.build()



class CondUNetTIMM(CondUnetBase):
    def __init__(self, in_depth, decoder_depths, last_conv_depth, downsample=nn.AvgPool2d,
                 upsample=nn.UpsamplingBilinear2d, cond_features=None, input_size=512, model_name='', **kwargs):
        super(CondUNetTIMM, self).__init__(in_depth, decoder_depths, last_conv_depth, downsample, upsample,
                                           cond_features, input_size, **kwargs)
        self.backbone = timm_backbones.get_model(model_name)
        self.down_depths, self.down_sample_list = timm_backbones.get_model_info(self.backbone)
        self.model_name = model_name
        if 'pnas' in model_name:
            self.down_depths = self.down_depths[:-1]
            self.down_sample_list = self.down_sample_list[:-1]
        self.build()

    def get_backbone_features(self, x, *args):

        if 'pnas' in self.model_name:
            backbone_features = self.backbone(x)[:-1]
        else:
            backbone_features = self.backbone(x)

        backbone_features = [x] + backbone_features
        return backbone_features


class BottleneckUp(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BottleneckUp, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv0 = conv3x3(inplanes, planes * self.expansion)
        self.conv1 = conv1x1(planes * self.expansion, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        x = self.conv0(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        if self.upsample is not None:
            out = self.upsample(out)

        return out


def get_cond_unet_model(in_depth=3, decoder_depths=(512, 256, 128, 64), last_conv_depth=6, downsample=nn.AvgPool2d,
                        upsample=nn.UpsamplingBilinear2d, cond_features=None, backbone='vgg', **kwargs):
    backbone = backbone.lower()
    if backbone == 'vgg':

        model = CondUNetVGG16(in_depth, decoder_depths, last_conv_depth, downsample, upsample,
                              cond_features, **kwargs)
    elif backbone == 'pnasnet5large':

        model = CondUNetTIMM(in_depth, decoder_depths, last_conv_depth, downsample, upsample,
                             cond_features, model_name=backbone, **kwargs)
    else:
        raise NotImplementedError()
    return model

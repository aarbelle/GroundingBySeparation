from torch import nn
import torch
import torch.nn.functional as torch_func
from networks import get_cond_unet_model
from transofmer_models import get_bert_model
from collections import OrderedDict


class CondModel(nn.Module):
    def __init__(self, last_conv_depth, decoder_depths, backbone='vgg', img2txt_weight=0, **kwargs):
        super(CondModel, self).__init__()
        self.caption_model = get_bert_model(out_features=None, average_outputs=True)
        cond_features = self.caption_model.get_depth() 
        self.image_model = get_cond_unet_model(decoder_depths=decoder_depths, last_conv_depth=last_conv_depth,
                                               cond_features=cond_features, backbone=backbone,
                                               **kwargs)
        self.out_down_factor = self.image_model.out_down_factor
        self.img2txt_weight = img2txt_weight

    def forward(self, *inputs, run_no_cond=1., use_neg=False):
        return self.forward_blended(*inputs, run_no_cond=run_no_cond, use_neg=use_neg)

    def forward_blended(self, *inputs, run_no_cond=1., use_neg=False):
        (input_image, input_caption1, input_caption2, image1, image2, caption_mask1, caption_mask2, mask,
         alpha) = inputs

        caption_embedding1 = self.caption_model(input_caption1, mask=caption_mask1)
        caption_embedding2 = self.caption_model(input_caption2, mask=caption_mask2)

        if not use_neg:

            outputs = self.image_model(input_image, caption_embedding1, caption_embedding2,
                                       run_no_cond=run_no_cond)
            output1 = outputs[0]
            output2 = outputs[1]
            output_no_c = outputs[-1] if run_no_cond else None
            output_neg = None

        else:
            caption_embedding_neg = torch.roll(caption_embedding2, 1, dims=0)
            outputs = self.image_model(input_image, caption_embedding1, caption_embedding2, caption_embedding_neg,
                                       run_no_cond=run_no_cond)
            output1 = outputs[0]
            output2 = outputs[1]
            output_neg = outputs[2]
            output_no_c = outputs[-1] if run_no_cond else None

        # img2txt loss
        if self.img2txt_weight:
            backbone_features = self.image_model.get_backbone_features(torch.cat((image1, image2), dim=0))
            e_w_mg1 = self.caption_model(input_caption1, mask=caption_mask1, average_outputs=False)
            e_w_mg2 = self.caption_model(input_caption2, mask=caption_mask2, average_outputs=False)
            e_w_mg_tuple = (e_w_mg1, e_w_mg2)
            bigger_1 = 0 if e_w_mg1.shape[1] > e_w_mg2.shape[1] else 1
            zero_padded_w_mg = torch.zeros_like(e_w_mg_tuple[bigger_1])
            zero_padded_w_mg[:, :e_w_mg_tuple[1-bigger_1].shape[1], :] = e_w_mg_tuple[1 - bigger_1]
            if bigger_1 == 0:
                e_w_mg2 = zero_padded_w_mg
            else:
                e_w_mg1 = zero_padded_w_mg
            sen_embedding = torch.cat((caption_embedding1, caption_embedding2), dim=0)
            word_embedding = torch.cat((e_w_mg1, e_w_mg2), dim=0)
            pre_cond_features = \
                [pre_cond_layer(feature) for feature, pre_cond_layer in zip(backbone_features[1:],
                                                                            self.image_model.pre_cond)]
            resized_pre_cond_features = [self.image_model.scale_conv(feature) for feature in pre_cond_features]
            img_embedding = torch.cat([pre_cond.unsqueeze(0) for pre_cond in resized_pre_cond_features], dim=0
                                      ).reshape((len(resized_pre_cond_features), resized_pre_cond_features[0].shape[0],
                                                 resized_pre_cond_features[0].shape[1],
                                                 resized_pre_cond_features[0].shape[-1]**2)).permute((1, 3, 0, 2))
            loss_img2txt_w, loss_img2txt_s = self.forward_img2txt(img_embedding, word_embedding, sen_embedding)
            loss_img2txt_w *= self.img2txt_weight
            loss_img2txt_s *= self.img2txt_weight

        else:
            loss_img2txt_w, loss_img2txt_s = 2 * [None]

        return output1, output2, output_neg, output_no_c, loss_img2txt_w, loss_img2txt_s

    @staticmethod
    def forward_img2txt(image_embedding, word_embedding, sentence_embedding):
        sentence_embedding = torch_func.normalize(sentence_embedding, dim=-1)
        word_embedding = torch_func.normalize(word_embedding, dim=-1)
        image_embedding = torch_func.normalize(image_embedding, dim=-1)
        heatmap_sentence_image = torch_func.relu(torch.einsum('bj,cklj->bckl', sentence_embedding, image_embedding))

        attention_sentence_image = torch.matmul(heatmap_sentence_image.permute(1,3,0,2),
                                                image_embedding.permute(0, 2, 1, 3)).permute(2, 0, 3, 1)
        attention_sentence_image = torch_func.normalize(attention_sentence_image, dim=2)
        attention_sentence_image_sim = torch.einsum('bclk,bl->bck', attention_sentence_image, sentence_embedding)
        attention_sentence_image_sim_max = torch.max(attention_sentence_image_sim, dim=-1)[0]
        attention_sentence_image_sim_score = torch.diagonal(torch_func.softmax(10.0 * attention_sentence_image_sim_max,
                                                                               dim=0))
        attention_image_sentence_sim_score =  torch.diagonal(torch_func.softmax(10.0 * attention_sentence_image_sim_max,
                                                                                dim=1))
        sentence_image_loss = -torch.mean(torch.log(attention_sentence_image_sim_score))
        image_sentence_loss = -torch.mean(torch.log(attention_image_sentence_sim_score))
        heatmap_word_image = torch_func.relu(torch.einsum('bij,cklj->bcikl', word_embedding, image_embedding))
        attention_word_image = torch.einsum('bcijl,cjlk->bcikl', heatmap_word_image, image_embedding)
        attention_word_image = torch_func.normalize(attention_word_image, dim=3)
        attention_word_image_sim = torch.einsum('bcilk,bil->bcik', attention_word_image, word_embedding)
        attention_word_image_sim_max = torch.max(attention_word_image_sim, dim=-1)[0]
        attention_word_image_pairs = torch.log(torch.pow(torch.sum(torch.exp(5.0 * attention_word_image_sim_max),
                                                                   dim=2), 1/5.0))
        attention_word_image_sim_score = torch.diagonal(torch_func.softmax(10.0 * attention_word_image_pairs, dim=0))
        attention_image_word_sim_score = torch.diagonal(torch_func.softmax(10.0 * attention_word_image_pairs, dim=1))
        word_image_loss = -torch.mean(torch.log(attention_word_image_sim_score))
        image_word_loss = -torch.mean(torch.log(attention_image_word_sim_score))
        word_losses = word_image_loss + image_word_loss
        sentence_losses = sentence_image_loss + image_sentence_loss
        return word_losses, sentence_losses


class BaseTrainModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='none')

    def calc_loss(self, im1: torch.Tensor, im2: torch.Tensor, mask: torch.Tensor):

        im1_v = torch.flatten(im1, 1)
        im2_v = torch.flatten(im2, 1)
        mask_v = torch.flatten(mask, 1).detach()
        loss_image = self.criterion(im1_v, im2_v) * mask_v
        loss = loss_image.sum(dim=1) / mask_v.sum(dim=1)
        loss = loss.mean()
        loss_image = torch.reshape(loss_image, im1.shape)
        return loss, loss_image

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class CondAlphaMapTM(BaseTrainModule):
    def __init__(self, decoder_depths, backbone='vgg', img2txt_weight=0, **kwargs):
        super().__init__()
        last_conv_depth = 1
        self.decoder_depths = decoder_depths
        self.model = CondModel(decoder_depths=decoder_depths, last_conv_depth=last_conv_depth, backbone=backbone,
                               img2txt_weight=img2txt_weight, **kwargs)
        self.downsample = lambda x: torch.nn.functional.interpolate(x, scale_factor=1 / self.model.out_down_factor)
        self.upsample = lambda x: torch.nn.functional.interpolate(x, scale_factor=self.model.out_down_factor)
        self.img2txt_weight = img2txt_weight
        self.vis_scalars = OrderedDict({})
        self.vis_images = OrderedDict({})

    def forward(self, blended_inputs, run_no_cond=1., use_neg=1.):
        loss = 0
        vis_scalars = OrderedDict({})
        vis_images = OrderedDict({})
        if blended_inputs is not None:
            loss, vis_scalars, vis_images = self.forward_blened_image(*blended_inputs,
                                                                      run_no_cond=run_no_cond,
                                                                      use_neg=use_neg)
        self.vis_scalars = vis_scalars
        self.vis_images = vis_images

        return loss

    def forward_blened_image(self, *inputs, run_no_cond=1., use_neg=1.):
        (input_image, input_caption1, input_caption2, gt_image1, gt_image2, caption_mask1, caption_mask2, mask,
         alpha, alpha1, alpha2, text1, text2) = inputs
        vis_scalars = {}
        model_outputs = self.model(*inputs[:-4], run_no_cond=run_no_cond, use_neg=use_neg)
        heatmap1, heatmap2, heatmap_neg, heatmap_no_cond, loss_img2txt_w, loss_img2txt_s = model_outputs
        if alpha1 is None:
            target1_disp = alpha * mask
            target2_disp = (1 - alpha) * mask
            if self.model.out_down_factor > 1:
                mask = self.downsample(mask)
                alpha = self.downsample(alpha)

            target1 = alpha
            target2 = (1 - alpha)
        else:
            target1_disp = alpha1 * mask
            target2_disp = alpha2 * mask
            if self.model.out_down_factor > 1:
                mask = self.downsample(mask)
                alpha1 = self.downsample(alpha1)
                alpha2 = self.downsample(alpha2)

            target1 = alpha1
            target2 = alpha2

        loss_sep1, _ = self.calc_loss(heatmap1, target1, mask)
        loss_sep2, _ = self.calc_loss(heatmap2, target2, mask)

        if heatmap_no_cond is not None:
            loss_adv = self.calc_loss(heatmap_no_cond, 0.5*torch.ones_like(target1, device=target1.device), mask)[0]
            loss_adv.unsqueeze(0)
        else:
            loss_adv = 0
        if heatmap_neg is not None:
            loss_neg = self.calc_loss(heatmap_neg, 0.*torch.ones_like(target1, device=target1.device),
                                      mask)[0].unsqueeze(0)
        else:
            loss_neg = 0
        loss_sep1 = loss_sep1.unsqueeze(0)
        loss_sep2 = loss_sep2.unsqueeze(0)
        loss_gbs = loss_sep1 + loss_sep2 + run_no_cond*loss_adv + use_neg*loss_neg
        if loss_img2txt_w is not None:
            loss = loss_gbs + loss_img2txt_w + loss_img2txt_s
        else:
            loss = loss_gbs

        if torch.isnan(loss.squeeze()):
            print('Got Nan')
        vis_scalars['loss'] = loss.detach()
        vis_scalars['loss_gbs'] = loss_gbs.detach()
        vis_scalars['loss_sep1'] = loss_sep1.detach()
        vis_scalars['loss_sep2'] = loss_sep2.detach()

        if heatmap_no_cond is not None:
            vis_scalars['loss_adv'] = loss_adv.detach()

        if heatmap_neg is not None:
            vis_scalars['loss_neg'] = loss_neg.detach()
        if not loss_img2txt_w is None:
            vis_scalars['loss_img2txt_w'] = loss_img2txt_w
            vis_scalars['loss_img2txt_s'] = loss_img2txt_s
        vis_images = OrderedDict({
            'Orig Images': (gt_image1[:1].detach(), gt_image2[:1].detach()),
            'Blended Image': input_image[:1].detach(),
            'Alpha': (target1_disp[:1].detach(), target2_disp[:1].detach()),
            'Network outputs': (self.upsample(heatmap1[:1].detach()), self.upsample(heatmap2[:1].detach())),

        })
        if heatmap_no_cond is not None:
            vis_images['No Cond'] = self.upsample(heatmap_no_cond[:1].detach())
        if heatmap_neg is not None:
            vis_images['Neg'] = self.upsample(heatmap_neg[:1].detach())
        return loss, vis_scalars, vis_images

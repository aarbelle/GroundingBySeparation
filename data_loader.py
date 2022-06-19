import traceback

import cv2
import pickle
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torch_data
import os
from PIL import Image, ImageOps
import numpy as np
from utils import TimeRecorder
from turbojpeg import TurboJPEG
import typing

from transofmer_models import get_bert_tokenizer
import generate_alpha as ga
import lmdb

tjpeg = TurboJPEG()


class BaseBatch(object):

    def pin_memory(self):
        for att_name in dir(self):
            att = getattr(self, att_name)
            if isinstance(att, torch.Tensor):
                att = att.pin_memory()
                setattr(self, att_name, att)
            elif isinstance(att, list):
                for item_num, item in enumerate(att):
                    if isinstance(item, torch.Tensor):
                        att[item_num] = item.pin_memory()

                setattr(self, att_name, att)
        return self

    def to(self, *args):
        for att_name in dir(self):
            att = getattr(self, att_name)
            if isinstance(att, torch.Tensor):
                att = att.to(*args)
                setattr(self, att_name, att)

            elif isinstance(att, list):
                for item_num, item in enumerate(att):
                    if isinstance(item, torch.Tensor):
                        att[item_num] = item.to(*args)
                setattr(self, att_name, att)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def detach(self):
        for att_name in dir(self):
            att = getattr(self, att_name)
            if isinstance(att, torch.Tensor):
                att = att.detach()
                setattr(self, att_name, att)
            elif isinstance(att, list):
                for item_num, item in enumerate(att):
                    if isinstance(item, torch.Tensor):
                        att[item_num] = item.detach()
                setattr(self, att_name, att)
        return self

    def numpy(self):
        self.to('cpu').detach()
        for att_name in dir(self):
            att = getattr(self, att_name)
            if isinstance(att, torch.Tensor):
                att = att.numpy()
                setattr(self, att_name, att)
            elif isinstance(att, list):
                for item_num, item in enumerate(att):
                    if isinstance(item, torch.Tensor):
                        att[item_num] = item.numpy()
                setattr(self, att_name, att)
        return self


class Resize(object):
    """
    This class resizes images to a square shape keeping aspect ration.
    The image is first resized such that the longer axis matches the desired size and then pads the short axis with
    zeros on each side.
    """

    def __init__(self, size: int, interpolation: int = None, fill_value: int = 0, ar_perserve=True,
                 resize_axis='long'):
        self.size = size
        self.interpolation = interpolation  # Defaults to BILINEAR Interpolation
        self.fill_value = fill_value
        self.ar_perserve = ar_perserve
        self.resize_axis = resize_axis

    def _call_ar_non_perserve_(self, img: typing.Union[np.ndarray, Image.Image]):
        if isinstance(img, np.ndarray):
            interpolation = self.interpolation if self.interpolation is not None else cv2.INTER_LINEAR
            old_size = list(reversed(img.shape[:2]))  # old_size is in (height, width) format
            depth = img.shape[2] if len(img.shape) > 1 else 1
        elif isinstance(img, Image.Image):
            interpolation = self.interpolation if self.interpolation is not None else Image.BILINEAR
            old_size = img.size  # old_size is in (width, height) format
            depth = len(img.getbands())
        else:
            raise TypeError('img should be of type PIL.Image.Image or numpy.ndarray but got {}'.format(type(img)))

        new_size = self.size

        mask = np.ones((self.size, self.size, depth), dtype=np.float32)

        if isinstance(img, np.ndarray):
            resized = cv2.resize(img, (new_size, new_size), interpolation=interpolation)



        elif isinstance(img, Image.Image):
            resized = img.resize(new_size, resample=self.interpolation)

        return resized, mask

    def _call_ar_perserve_(self, img: typing.Union[np.ndarray, Image.Image]):
        if isinstance(img, np.ndarray):
            interpolation = self.interpolation if self.interpolation is not None else cv2.INTER_LINEAR
            old_size = list(reversed(img.shape[:2]))  # old_size is in (height, width) format
            depth = img.shape[2] if len(img.shape) > 1 else 1
        elif isinstance(img, Image.Image):
            interpolation = self.interpolation if self.interpolation is not None else Image.BILINEAR
            old_size = img.size  # old_size is in (width, height) format
            depth = len(img.getbands())
        else:
            raise TypeError('img should be of type PIL.Image.Image or numpy.ndarray but got {}'.format(type(img)))

        ratio = float(self.size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        delta_w = self.size - new_size[0]
        delta_h = self.size - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

        mask = np.ones((new_size[1], new_size[0], depth), dtype=np.float32)
        padded_mask = cv2.copyMakeBorder(mask, padding[1], padding[3], padding[0], padding[2], cv2.BORDER_CONSTANT)
        if isinstance(img, np.ndarray):
            resized = cv2.resize(img, new_size, interpolation=interpolation)

            padded = cv2.copyMakeBorder(resized, padding[1], padding[3], padding[0], padding[2], cv2.BORDER_CONSTANT)
            return padded, padded_mask
        elif isinstance(img, Image.Image):
            resized = img.resize(new_size, resample=self.interpolation)

            return ImageOps.expand(resized, padding), padded_mask

    def __call__(self, img: typing.Union[np.ndarray, Image.Image]):
        if self.ar_perserve:
            return self._call_ar_perserve_(img)
        else:
            return self._call_ar_non_perserve_(img)


class LmdbDataset(torch_data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, dataset_path, split, resize_transform=None, transform=None, debug_timing=False,
                 max_len=None, caption_augs=None, is_training=True):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset_path: Root path to data.
            split: train/test/val.
            transform: image transformer.
        """
        self.dataset_path = dataset_path
        self.lmdb_filename = os.path.join(dataset_path, 'images.lmdb')
        self.annotations_pickle = os.path.join(dataset_path, 'annotations', f'{split}.pickle')
        self.raw_im_path = os.path.join(dataset_path, 'jpeg')
        self.debug_timing = debug_timing
        self.max_len = max_len
        self.caption_augs = caption_augs
        if 'visual_genome' in dataset_path:
            self.corrupt_list = VG_CORRUPT_LIST
        else:
            self.corrupt_list = []
        self.tokenizer = get_bert_tokenizer()

        if debug_timing:
            self.timer = TimeRecorder()
        else:
            self.timer = None
        with open(self.annotations_pickle, 'rb') as fp:
            self.annotations = pickle.load(fp, encoding='latin1')

        lmdb_env = lmdb.open(self.lmdb_filename, map_size=int(1e11), readonly=True, lock=False)
        self.txn = lmdb_env.begin(write=False)

        self.batch = None
        self.ann_ids = list(set(self.annotations.keys()))
        self.ids = []
        for ann_id in self.ann_ids:
            if is_training:
                if len(self.annotations[ann_id]['queries']) == 0 or (ann_id in self.corrupt_list):
                    continue
                for q_id, q in enumerate(self.annotations[ann_id]['queries']):

                    self.ids.append((ann_id, q_id))
            else:
                self.ids.append((ann_id, -1))

        self.ids.sort(key=lambda x: (x[0], x[1]))
        random.seed(0)
        random.shuffle(self.ids)
        self.transform = transform
        self.resize_transorm = resize_transform
        self.to_tensor_transform = torchvision.transforms.ToTensor()

    def sen2token(self, caption, do_augs=False):
        if not isinstance(caption, list):
            caption = caption.replace('’', "'").replace('\n', ' ').replace('´',"'")
            words = caption.split()
        else:
            words = [w.replace('’', "'").replace('\n', '').replace('´',"'")  for w in caption]
        if self.caption_augs is not None and do_augs:
            for aug in self.caption_augs:

                if aug.startswith('drop'):
                    rate = float(aug.split(':')[-1])
                    selected = np.random.rand(len(words)) >= rate
                    while not np.any(selected):
                        selected = np.random.rand(len(words)) >= rate
                    words = [w for w, s in zip(words, selected) if s]
                if aug == 'shuffle':
                    random.shuffle(words)

            caption = ' '.join(words)

        if self.max_len and len(words) > self.max_len:
            s = np.random.randint(0, len(words) - self.max_len)
            caption = ' '.join(words[s:s + self.max_len])
        caption_emb_ = self.tokenizer.encode(caption, add_special_tokens=True)
        token_mask = self.tokenizer.get_special_tokens_mask(caption_emb_, already_has_special_tokens=True)
        token = torch.tensor(caption_emb_)
        if len(token.shape) > 1:
            token_mask = [2 * (t == 0).all(dim=-1) + (m > 0) for m, t in zip(token_mask, token)]
        else:
            token_mask = [2 * (t == 0) + (m > 0) for m, t in zip(token_mask, token)]
        token_mask = torch.tensor(token_mask)
        return words, token, token_mask

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        try:
            if self.timer:
                self.timer.tic('Start getitem')
            ann_id, q_id = self.ids[index]
            ann_tensor = torch.tensor([int(ann_id), q_id])
            if self.timer:
                self.timer.toc('get path and annotations')

            if len(self.annotations[ann_id]['queries']) == 0:
                raise ValueError
            if os.path.exists(os.path.join(self.raw_im_path, f'{ann_id}.jpg')):
                # image = np.zeros((5,5,3))
                image = cv2.imread(os.path.join(self.raw_im_path, f'{ann_id}.jpg'))
                if image is None:
                    print(f'Could not read image {os.path.join(self.raw_im_path, f"{ann_id}.jpg")}')
                    imgbin = self.txn.get(ann_id.encode('utf-8'))
                    if imgbin is not None:
                        buff = np.frombuffer(imgbin, dtype='uint8')
                    else:
                        raise ValueError

                    image = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                    cv2.imwrite(os.path.join(self.raw_im_path, f'{ann_id}.jpg'), image)
            else:
                imgbin = self.txn.get(ann_id.encode('utf-8'))
                if imgbin is not None:
                    buff = np.frombuffer(imgbin, dtype='uint8')
                else:
                    raise ValueError
                image = cv2.imdecode(buff, cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(self.raw_im_path, f'{ann_id}.jpg'), image)
            # imgbin = self.txn.get(ann_id.encode('utf-8'))
            # if imgbin is not None:
            #     buff = np.frombuffer(imgbin, dtype='uint8')
            # else:
            #     raise ValueError
            # image = cv2.imdecode(buff, cv2.IMREAD_COLOR)
            orig_image_size = torch.tensor(image.shape[:2])
            if self.timer:
                self.timer.toc('load image')
            if self.resize_transorm:
                image, mask = self.resize_transorm(image)

                mask = self.to_tensor_transform(mask)

            if self.transform:
                image = self.transform(image)
            if not self.resize_transorm:
                mask = torch.ones_like(image)
            if self.timer:
                self.timer.toc('transform image')
                print(self.timer)
            caption = self.annotations[ann_id]['queries'][q_id] # This is the text
            caption_words, caption_tokens, cap_tokens_mask = self.sen2token(caption, do_augs=True)

            if self.timer:
                self.timer.toc('Embed Caption')
                print(self.timer)
            annid_qid = f'{ann_id},{q_id}'
            if q_id == -1:
                phrases_tok, phrases_tok_mask = self.get_tokens(ann_id, q_id)
            else:
                phrases_tok, phrases_tok_mask = self.get_tokens(ann_id, q_id)
            caption_words = np.array(caption_words).astype(np.str_)
        except Exception as err:
            if not (err in (KeyboardInterrupt, InterruptedError)):
                track = traceback.format_exc()
                print(track)
        return (image, mask, caption_tokens, cap_tokens_mask, caption_words, annid_qid, phrases_tok,
                phrases_tok_mask, orig_image_size, ann_tensor)

    def get_tokens(self, ann_id, q_id):
        phrases = []
        phrases_tokens = []
        phrases_tok_mask = []
        for phrase in self.annotations[ann_id]['annotations']:
            phrase.pop('segmentation', None)
            if 'query' in phrase.keys():
                src_sen = phrase.get('source_sen', phrase['query'])
                if q_id == -1 or (src_sen == self.annotations[ann_id]['queries'][q_id]):
                    q = phrase['query']
                else:
                    q = None
            else:

                q = phrase['category']
                phrase['query'] = q
                phrase['idx'] = [phrase['category_id']]
                phrase['bbox'] = [phrase['bbox']]
                phrase['category'] = [phrase['category']]
            if not q:
                continue
            words, tok, tok_mask = self.sen2token(q)

            # phrases.append(np.array(words).astype(np.str_))


            phrases_tokens.append(tok)
            phrases_tok_mask.append(tok_mask)
        phrases_tokens, phrases_tok_mask = self.norm_length(phrases_tokens, phrases_tok_mask)
        return phrases_tokens, phrases_tok_mask

    def get_annotations(self, annid_qid: str, orig_image_size):
        if isinstance(annid_qid, torch.Tensor):
            ann_id, q_id = annid_qid[0].item(), annid_qid[1].item()
            ann_id = str(ann_id)
        else:
            if isinstance(annid_qid, np.bytes_):
                annid_qid = annid_qid.decode('utf-8')
            ann_id, q_id = annid_qid.split(',')

        q_id = int(q_id)
        bboxes = []
        out_phrase = []
        for phrase in self.annotations[ann_id]['annotations']:
            phrase.pop('segmentation', None)
            if 'query' in phrase.keys():
                src_sen = phrase.get('source_sen', phrase['query'])
                if q_id == -1 or src_sen == self.annotations[ann_id]['queries'][q_id]:
                    q = phrase['query']
                else:
                    q = None
            else:

                q = phrase['category']
                phrase['query'] = q
                phrase['idx'] = [phrase['category_id']]
                phrase['bbox'] = [phrase['bbox']]
                phrase['category'] = [phrase['category']]
            if not q:
                continue

            bboxes.append(phrase['bbox'])
            if 'image_size' not in phrase.keys():
                phrase['image_size'] = orig_image_size
            out_phrase.append(phrase)
        return out_phrase, bboxes

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def norm_length(tokens, masks, max_l=0):

        if max_l == 0:
            for token in tokens:
                # noinspection PyUnusedLocal
                max_l = max(len(token), max_l)
        if isinstance(tokens, tuple):
            tokens = list(tokens)
        if isinstance(masks, tuple):
            masks = list(masks)

        for ind, (token, mask) in enumerate(zip(tokens, masks)):
            pad_len = max_l - len(token)

            if pad_len > 0:

                masks[ind] = torch.cat((mask, 2 * torch.ones(pad_len, dtype=mask.dtype)), dim=0)

                if isinstance(token, torch.Tensor) and len(token.shape) > 0:
                    pad_len = [pad_len] + list(token[0].shape)
                tokens[ind] = torch.cat((token, torch.zeros(pad_len, dtype=token.dtype)), dim=0)
        if not len(tokens):
            return tokens, masks
        return torch.stack(tokens, dim=0), torch.stack(masks, dim=0)

    @classmethod
    def collate_fn(cls, tokenizer: transforms, debug_timing=False):
        if debug_timing:
            timer = TimeRecorder()

        # noinspection PyUnreachableCode
        def fun(data):
            if debug_timing:
                timer.tic('Start Collate Function')

            (img, mask, full_cap_tokens, full_cap_masks, cap_words, ann_id, queries_tok, q_tok_emb_mask,
             orig_image_size, ann_tensor) = zip(*data)

            q_emb_sorted = sorted(queries_tok, key=lambda x: len(x))
            max_phrases = len(q_emb_sorted[-1])
            caps = [[] for _ in range(max_phrases)]
            caps_masks = [[] for _ in range(max_phrases)]
            max_len = 0
            for c_ind, (cap_emb, cap_mask) in enumerate(zip(queries_tok, q_tok_emb_mask)):
                for q_ind, (tok, q_mask) in enumerate(zip(cap_emb, cap_mask)):
                    caps[q_ind].append(tok)
                    caps_masks[q_ind].append(q_mask)
                    max_len = max(max_len, len(tok))
                for q_ind in range(len(cap_emb), max_phrases):
                    emp, emp_mask = tokenizer.get_empty_token()
                    caps[q_ind].append(emp)
                    caps_masks[q_ind].append(emp_mask)

            all_emb = []
            all_masks = []
            for queries_, q_masks in zip(caps, caps_masks):
                queries_tok, q_mask = cls.norm_length(queries_, q_masks, max_len)
                all_emb.append(queries_tok)
                all_masks.append(q_mask)
            img = torch.stack(img, 0)
            mask = torch.stack(mask, 0)
            full_cap_tokens, full_cap_masks = cls.norm_length(full_cap_tokens, full_cap_masks)
            if debug_timing:
                timer.toc('Zip Data')
                print(timer)

            return cls.Batch(img, mask, torch.stack(all_emb), torch.stack(all_masks),
                             torch.stack(ann_tensor), torch.stack(orig_image_size),
                             full_cap_tokens, full_cap_masks)

        return fun

    class Batch(BaseBatch):
        def __init__(self, image, mask, caption_emb,
                     caption_mask, ann_id, image_size,
                     full_cap_tokens, full_cap_masks):
            """
            :param image:
            :type  image: torch.Tensor
            :param mask:
            :type  mask:torch.Tensor]
            :param caption_emb:
            :type  caption_emb:  List[torch.Tensor]
            :param caption_mask:
            :type  caption_mask:  List[torch.Tensor]
            :param captions:
            :type  captions: List[str]
            :param ann_id:
            :type  ann_id: int
            :param annotations:
            :param boxes:
            :type  boxes: list
            """
            self.image = image
            self.mask = mask
            self.caption_emb = caption_emb

            self.caption_mask = caption_mask

            self.ann_id = ann_id
            self.image_size = image_size
            self.full_cap_tokens = full_cap_tokens
            self.full_cap_masks = full_cap_masks


class LmdbPairDataset(LmdbDataset):

    @classmethod
    def collate_fn(cls, tokenizer: transforms, debug_timing=False, alpha_map=1, seed=None, alpha_downscale=1):

        if debug_timing:
            timer = TimeRecorder()
        if not isinstance(alpha_map, list):
            alpha_types = [alpha_map]
            alpha_percents = np.array([1.])
        elif len(alpha_map) == 1:
            alpha_types = alpha_map
            alpha_percents = np.array([1.])
        else:

            if len(alpha_map) % 2:
                raise ValueError(f'alpha map should be a list of even length! got alpha map list of length '
                                 f'{len(alpha_map)}')
            alpha_types = alpha_map[::2]
            alpha_percents = np.array(alpha_map[1::2])

            if not sum(alpha_percents) == 1:
                raise ValueError(f'alpha percentages should sum to 1. '
                                 f'Got sum({list(alpha_percents)}) = {alpha_percents.sum()}')

        def fun(data):

            if debug_timing:
                timer.tic('Start Collate Function')
            num_samples = int(len(data) / 2)
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            rank = int(os.environ.get('RANK', 0))
            total_num_samples = num_samples*world_size
            num_sample_list = np.round(alpha_percents*total_num_samples).astype(np.int)
            num_sample_list[-1] = num_sample_list[-1] + (total_num_samples-num_sample_list.sum())
            start_sample = rank*num_samples
            stop_sample = (rank+1)*num_samples
            running_count = 0
            for i, alpha_sample in enumerate(num_sample_list):
                if running_count >= stop_sample:
                    num_sample_list[i] = 0
                    running_count += alpha_sample
                    continue
                if running_count + alpha_sample < start_sample:
                    running_count += alpha_sample
                    num_sample_list[i] = 0
                    continue
                if running_count < start_sample:
                    alpha_sample = alpha_sample - (start_sample - running_count)
                    running_count = start_sample
                new_alpha_sample = min(alpha_sample, stop_sample-running_count)
                running_count += alpha_sample

                num_sample_list[i] = new_alpha_sample

            img1, mask1, cap_emb1, cap_emb_mask1, cap1, ann_id1, _, _, _, _ = zip(*data[:num_samples])
            img2, mask2, cap_emb2, cap_emb_mask2, cap2, ann_id2, _, _, _, _ = zip(*data[num_samples:2 * num_samples])
            try:
                rng = np.random.RandomState(np.uint32(ann_id1[0].split(',')[0]) + np.uint32(ann_id2[0].split(',')[0]))

                cap_emb, cap_mask = cls.norm_length(cap_emb1 + cap_emb2, cap_emb_mask1 + cap_emb_mask2)
                cap_emb1, cap_emb2 = cap_emb[:num_samples], cap_emb[num_samples:2 * num_samples]
                cap_mask1, cap_mask2 = cap_mask[:num_samples], cap_mask[num_samples:2 * num_samples]
                alpha_lut = {1: 'perlin', 2: 'normed_gaussian'}
                img1_orig = img1
                mask1_orig = mask1
                img2_orig = img2
                mask2_orig = mask2
                all_alpha = []
                all_alpha1 = []
                all_alpha2 = []
                all_img1 = []
                all_img2 = []
                start_sample = 0

                for alpha_mode, alpha_num_samp in zip(alpha_types, num_sample_list):
                    if alpha_num_samp == 0:
                        continue
                    if not alpha_num_samp:
                        continue
                    img1 = img1_orig[start_sample:start_sample+alpha_num_samp]
                    mask1 = mask1_orig[start_sample:start_sample+alpha_num_samp]
                    img2 = img2_orig[start_sample:start_sample+alpha_num_samp]
                    mask2 = mask2_orig[start_sample:start_sample+alpha_num_samp]
                    alpha1 = alpha2 = None
                    start_sample += alpha_num_samp
                    if int(alpha_mode) not in alpha_lut.keys():
                        raise NotImplementedError
                    blnd_alpha_mode = alpha_lut[int(alpha_mode)]

                    num_seeds = 1
                    _, height, width = img1[0].shape

                    if blnd_alpha_mode == 'perlin':
                        mask = [m1 * m2 for m1, m2 in zip(mask1, mask2)]
                        max_seeds = 2
                        num_seeds = rng.randint(1, max_seeds+1)
                        seeds = rng.choice(np.arange(max_seeds), num_seeds)
                        alpha = ga.get_perlin_alpha(height, width, alpha_downscale, rng, seeds, mask)
                    elif blnd_alpha_mode == 'normed_gaussian':
                        alpha1 = ga.get_gmm_alpha(alpha_num_samp, height, width, num_seeds, rng=rng)
                        alpha2 = ga.get_gmm_alpha(alpha_num_samp, height, width, num_seeds, rng=rng)
                        alpha = alpha1/(alpha1 + alpha2 + 1e-7)
                    else:
                        raise NotImplementedError(f'Unsupported blnd_alpha_mode={blnd_alpha_mode}')
                    if alpha1 is None:
                        alpha1 = alpha
                        alpha2 = 1-alpha1

                    all_alpha.append(alpha)
                    all_alpha1.append(alpha1)
                    all_alpha2.append(alpha2)
                    all_img1.extend(img1)
                    all_img2.extend(img2)
                if len(all_alpha) > 1:
                    alpha = torch.cat(all_alpha, dim=0)
                    alpha1 = torch.cat(all_alpha1, dim=0)
                    alpha2 = torch.cat(all_alpha2, dim=0)
                else:
                    alpha = all_alpha[0]
                    alpha1 = all_alpha1[0]
                    alpha2 = all_alpha2[0]

                img1 = torch.stack(all_img1, 0)
                img2 = torch.stack(all_img2, 0)
                mask1 = torch.stack(mask1_orig, 0)
                mask2 = torch.stack(mask2_orig, 0)
                if debug_timing:
                    timer.toc('Zip Data')
                    print(timer)

            except Exception as err:
                print('Problem with batch')
                print(f'AnnId1: {ann_id1}')
                print(f'AnnId2: {ann_id2}')
                if not (err in (KeyboardInterrupt, InterruptedError)):
                    track = traceback.format_exc()
                    print(track)

            return cls.Batch(img1, img2, mask1, mask2, alpha, alpha1, alpha2, cap_emb1, cap_emb2, cap_mask1,
                             cap_mask2,
                             np.array(cap1), np.array(cap2), np.array(ann_id1).astype(np.str_),
                             np.array(ann_id2).astype(np.str_))

        return fun

    class Batch(BaseBatch):
        def __init__(self, image1, image2, mask1, mask2, alpha, alpha1, alpha2, caption_emb1, caption_emb2,
                     caption_mask1, caption_mask2, captions1,
                     captions2,
                     ann_id1, ann_id2):
            """
            :param image1:
            :type  image1: torch.Tensor
            :param image2:
            :type  image2: torch.Tensor
            :param mask1:
            :type  mask1: torch.Tensor
            :param mask2:
            :type  mask2: torch.Tensor
            :param alpha:
            :type  alpha: torch.Tensor
            :param caption_emb1:
            :type  caption_emb1: torch.Tensor
            :param caption_emb2:
            :type  caption_emb2: torch.Tensor
            :param caption_mask1:
            :type  caption_mask1: torch.Tensor
            :param caption_mask2:
            :type  caption_mask2: torch.Tensor
            :param captions1:
            :type  captions1: str
            :param captions2:
            :type  captions2: str
            :param ann_id1:
            :type  ann_id1: int
            :param ann_id2:
            :type  ann_id2: int
            """
            self.image1 = image1
            self.image2 = image2
            self.mask1 = mask1
            self.mask2 = mask2
            self.alpha = alpha
            self.alpha1 = alpha1
            self.alpha2 = alpha2
            self.caption_emb1 = caption_emb1
            self.caption_emb2 = caption_emb2
            self.caption_mask1 = caption_mask1
            self.caption_mask2 = caption_mask2
            self.captions1 = captions1
            self.captions2 = captions2
            self.ann_id1 = ann_id1
            self.ann_id2 = ann_id2


def get_lmdb_image_pair_loader(images_root, split, resize_transform, transform, batch_size, num_workers, max_len=None,
                               alpha_map=False, debug_timing=False, alpha_downscale=1, drop_last=False, shuffle=True,
                               pin_memory=True, seed=None, caption_augs=None,
                               world_size=None, rank=None) -> typing.Iterator[LmdbPairDataset.Batch]:
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset

    if isinstance(images_root, str):
        readers = [LmdbPairDataset(dataset_path=images_root, split=split, resize_transform=resize_transform,
                                   transform=transform, max_len=max_len, debug_timing=debug_timing,
                                   caption_augs=caption_augs)]
    elif len(images_root) == 1:
        images_root = images_root[0]
        readers = [LmdbPairDataset(dataset_path=images_root, split=split, resize_transform=resize_transform,
                                   transform=transform, max_len=max_len, debug_timing=debug_timing,
                                   caption_augs=caption_augs)]
    else:

        readers = [LmdbPairDataset(dataset_path=r, split=split, resize_transform=resize_transform,
                                   transform=transform, max_len=max_len, debug_timing=debug_timing,
                                   caption_augs=caption_augs) for r in
                   images_root]
    reader = torch.utils.data.ConcatDataset(readers)

    sampler = torch.utils.data.distributed.DistributedSampler(
        reader,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    shuffle = False

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=reader,
                                              batch_size=batch_size * 2,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              collate_fn=readers[0].collate_fn(readers[0].tokenizer,
                                                                               debug_timing=debug_timing,
                                                                               alpha_map=alpha_map, seed=seed,
                                                                               alpha_downscale=alpha_downscale),
                                              drop_last=drop_last,
                                              sampler=sampler)
    # noinspection PyTypeChecker
    return data_loader


def get_bert_image_query_loader(data_root, split, resize_transform, transform, batch_size, num_workers,
                                max_len=None, debug_timing=False, drop_last=False, shuffle=True, pin_memory=True,
                                world_size=1, rank=0) -> typing.Iterator[LmdbDataset.Batch]:
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    reader = LmdbDataset(dataset_path=data_root, split=split, resize_transform=resize_transform, transform=transform,
                         max_len=max_len, debug_timing=debug_timing, is_training=True)
    sampler = torch.utils.data.distributed.DistributedSampler(
        reader,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    shuffle = False
    data_loader = torch.utils.data.DataLoader(dataset=reader,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              collate_fn=reader.collate_fn(reader.tokenizer, debug_timing=debug_timing),
                                              drop_last=drop_last)
    # noinspection PyTypeChecker
    return data_loader, reader

# List of corrupt images from the visual genome training set. These images have no affect on the test.
VG_CORRUPT_LIST = [
"2386339",
"2386421",
"2417886",
"2384303",
"2399366",
"2387773",
"2388085",
"2388284",
"2388568",
"2388917",
"2388989",
"2389915",
"2390058",
"2390387",
"2329477",
"2316003",
"2316472",
"2321571",
"2318215",
"2319311",
"2319788",
"2320098",
"2320630",
"2321051",
"2323189",
"2323362",
"2324980",
"2325435",
"2325661",
"2325625",
"2326141",
"2327203",
"2329481",
"2328471",
"2328613",
"2330142",
"2331541",
"2335434",
"2333919",
"2335839",
"2335721",
"2335991",
"2336131",
"2339038",
"2336771",
"2336852",
"2337155",
"2339452",
"2337005",
"2338403",
"2339625",
"2340300",
"2340698",
"2343246",
"2343497",
"2344194",
"2344765",
"2344844",
"2345238",
"2345400",
"2346562",
"2346745",
"2346773",
"2347067",
"2348472",
"2350125",
"2350190",
"2350687",
"2351122",
"2355831",
"2354293",
"2354396",
"2356194",
"2356717",
"2357002",
"2357151",
"2357389",
"2357595",
"2359672",
"2360334",
"2360452",
"2361256",
"2363899",
"2368946",
"2369047",
"2369711",
"2369516",
"2370307",
"2370333",
"2373542",
"2370544",
"2370885",
"2376553",
"2370916",
"2372979",
"2374336",
"2374775",
"2400248",
"2385744",
"2386269",
"2400944",
"2401212",
"2378352",
"2401793",
"2378996",
"2379210",
"2391568",
"2379468",
"2402595",
"2379781",
"2379927",
"2413483",
"2392406",
"2380334",
"2392550",
"2392657",
"2380814",
"2414491",
"2393186",
"2393276",
"2414917",
"2415090",
"2404941",
"2405004",
"2405111",
"2405254",
"2394503",
"2416204",
"2416218",
"2405946",
"2382669",
"2406223",
"2394940",
"2416776",
"2383215",
"2383685",
"2407333",
"2407337",
"2407779",
"2407937",
"2408234",
"2408461",
"2396395",
"2396732",
"2397424",
"2408825",
"2397721",
"2397804",
"2410274",
"2410589",
"2395307",
"2412350",
"2417802",
"2395669",
"2387074",
"2394180",
"2354486",
"2344903",
"2403972",
"2385366",
"2405591",
"2315674",
"2414809",
"2338884",
"2390322",
"2407778",
"2400491",
"2415989",
"2402944",
"2354437",
]
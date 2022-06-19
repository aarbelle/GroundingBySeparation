import torch.distributed as dist
import numpy as np
import torch
import os
import evaluate_benchmarks_utils.utils as eval_utils
import concurrent.futures
from utils import logprint
from torch.nn.functional import interpolate


class GBSEvaluator(object):
	def __init__(self):

		self.cnt_overall = 0
		self.cnt_correct_hit = 0
		self.ranks = []
		self.cat_cnt_overall = {}
		self.cat_cnt_correct_hit = {}
		self.cat_cnt_correct_hit_normed = {}
		self.cnt = 0
		self.hit_acc = None

	@staticmethod
	def validate_instance(heatmap, annot):
		heatmap = heatmap
		hit_c = 0
		cat_cnt_overall = {}
		cat_cnt_correct_hit = {}

		orig_img_shape = annot['image_size'][:2]
		query = annot['query']
		idx = annot['idx']
		if len(query.split()) == 0 or len(idx) == 0:
			print('query zero')
		category = list(annot['category'])
		# skip non-groundable queries
		if 'notvisual' in category or len(annot['bbox_norm']) == 0:
			return 0, hit_c, cat_cnt_overall, cat_cnt_correct_hit
		if not eval_utils.check_percent(eval_utils.union(annot['bbox_norm'])):
			return 0, hit_c, cat_cnt_overall, cat_cnt_correct_hit
		for cat in category:
			if cat not in cat_cnt_overall:
				cat_cnt_overall[cat] = 0
			cat_cnt_overall[cat] += 1
		hit_c = eval_utils.calc_correctness(annot, heatmap, orig_img_shape)
		for cat in category:
			# per-cat hit acc
			if cat not in cat_cnt_correct_hit:
				cat_cnt_correct_hit[cat] = 0
			cat_cnt_correct_hit[cat] += hit_c

		return 1, hit_c, cat_cnt_overall, cat_cnt_correct_hit

	@staticmethod
	def resize_and_crop(hm, an):
		orig_size = np.array(an['image_size'])
		max_size = np.max(orig_size)
		hm_resize = interpolate(hm.unsqueeze(0).unsqueeze(0), (max_size, max_size), mode='bilinear')
		offset = (max_size - orig_size)//2
		hm_resize = hm_resize[0, 0, offset[0]:offset[0]+orig_size[0], offset[1]:offset[1]+orig_size[1]]
		return hm_resize

	def validate_batch(self, heatmaps, annotations, arp_resize=True, box_masks=None):
		threads = []
		multi_thread = True
		if box_masks is not None:
			box_masks = box_masks.unsqueeze(1)

		with concurrent.futures.ThreadPoolExecutor() as executor:
			for b_ind, anns in enumerate(annotations):
				for ann_num, annot in enumerate(anns):

					heatmap = heatmaps[ann_num][b_ind, 0]
					heatmap[:4, :] = 0
					heatmap[:, :4] = 0
					heatmap[-3:, :] = 0
					heatmap[:, -3:] = 0
					if arp_resize:
						heatmap = self.resize_and_crop(heatmap, annot)
					if box_masks is not None:
						mask = box_masks[b_ind:b_ind+1]
						mask = torch.nn.functional.interpolate(mask, size=heatmap.shape, mode='nearest')
						heatmap = heatmap*mask[0, 0]

					if multi_thread:
						threads.append(executor.submit(self.validate_instance, heatmap, annot))
					else:
						threads.append(self.validate_instance(heatmap, annot))
			for t in threads:
				if multi_thread:
					(overall_count, hit_c, cat_cnt_overall, cat_cnt_correct_hit) = t.result()
				else:
					(overall_count, hit_c, cat_cnt_overall, cat_cnt_correct_hit) = t
				self.cnt_overall += overall_count

				if overall_count:
					self.cnt_correct_hit += hit_c

				for k in cat_cnt_overall.keys():
					if k not in self.cat_cnt_overall.keys():
						self.cat_cnt_overall[k] = 0
						self.cat_cnt_correct_hit[k] = 0

					self.cat_cnt_overall[k] += cat_cnt_overall[k]
					self.cat_cnt_correct_hit[k] += cat_cnt_correct_hit[k]

	@staticmethod
	def gather_tensor(tensor):
		world_size = int(os.environ.get('WORLD_SIZE', 1))
		rank = int(os.environ.get('RANK', 0))
		shape = tensor.shape
		shape_list = [shape]*world_size

		if len(shape) == 1:
			max_shape = shape[0]
			shape_list = [tensor.new_ones(()) * rank for _ in range(world_size)]
			torch.distributed.all_gather(shape_list, tensor.new_ones(())*tensor.shape[0], async_op=False)
			for s in shape_list:
				if s >= max_shape:
					max_shape = s
			if shape[0] <= max_shape:
				orig_tensor = tensor
				tensor = orig_tensor.new_ones(max_shape)*np.nan
				tensor[:shape[0]] = orig_tensor

		elif len(shape) > 1:
			raise RuntimeError('Unsuporrted tensors of high demensions')
		else:
			max_shape = shape
		gather_list = [tensor.new_ones(max_shape) for _ in shape_list]
		torch.distributed.all_gather(gather_list, tensor, async_op=False)
		for ind, g in enumerate(gather_list):
			gather_list[ind] = g[torch.logical_not(torch.isnan(g))]
		try:
			gatherd = torch.cat(gather_list, dim=0)
		except RuntimeError:
			gatherd = torch.stack(gather_list, dim=0)

		return gatherd

	def gather_results(self, device):
		self.cnt_correct_hit = torch.tensor(self.cnt_correct_hit).to(device)
		self.cnt_overall = torch.tensor(self.cnt_overall).to(device)

		if int(os.environ.get('RANK', 0)) == 0 or True:
			self.cnt_correct_hit = self.gather_tensor(self.cnt_correct_hit).sum().float()
			self.cnt_overall = self.gather_tensor(self.cnt_overall).sum().float()

	def calc_results(self):
		# overall acc
		self.hit_acc = self.cnt_correct_hit / max(self.cnt_overall, 1)

		# cat-wise acc
		for cat in self.cat_cnt_correct_hit:
			self.cat_cnt_correct_hit_normed[cat] = self.cat_cnt_correct_hit[cat] / max(1, self.cat_cnt_overall[cat])

		return

	def print_results(self, heatmap_naming='', output_file=None):
		dist_rank = int(os.environ.get('RANK', 0))
		if dist_rank > 0:
			return
		lines_to_print = [f'Overall {heatmap_naming} Results:\nHit Accuracy: \t\t {self.hit_acc:0.4f}']
		max_len = len(lines_to_print[-1])

		# cat-wise acc
		for cat in self.cat_cnt_correct_hit:
			lines_to_print.append(f'{cat}:\nHit Accuracy: \t\t {self.cat_cnt_correct_hit_normed[cat]:0.4f}')
			max_len = max(max_len, len(lines_to_print[-1]))

		print('-' * max_len)
		for line in reversed(lines_to_print):
			print(line)
			print('-'*max_len)
		try:
			if output_file:
				with open(output_file, 'w') as fp:
					for line in reversed(lines_to_print):
						fp.write(line+'\n')
						fp.write('-' * max_len + '\n')
				os.umask(0)
				os.chmod(output_file, 0o777)
		except PermissionError:
			logprint(f'Permission Error: Could not save output file to {output_file}. ')

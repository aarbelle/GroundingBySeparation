import os
import traceback
import torch
import utils
import numpy as np
import torchvision.transforms as transforms
import modules_test as tm
from utils import Logger, logprint
from data_loader import Resize, get_bert_image_query_loader
from eval_benchmark_parameters import get_params, Params
from evaluate_benchmarks_utils.evaluators import GBSEvaluator
from tqdm import tqdm
import torch.distributed as dist
utils.set_all_random_seeds()

rng = np.random.RandomState(0)


def forward_pass(data, eval_module, params):
    data = data.to('cuda')
    # prepare inputs:
    image = data.image
    mask = data.mask[:, :1]
    inputs = [image]
    for emb, cap_mask in zip(data.caption_emb, data.caption_mask):
        inputs.extend([emb, cap_mask])
    inputs.extend([data.full_cap_tokens, data.full_cap_masks])

    # Forward pass
    heatmaps, img2txt_heatmaps = eval_module(*inputs)

    masked_heatmaps = tuple((torch.nn.functional.interpolate(hm, mask.shape[2:4],
                                                             mode='nearest') * mask) for hm in heatmaps)
    gbs_heatmaps = tuple((torch.nn.functional.interpolate(mask, hm.shape[2:4],
                                                          mode='bilinear') * hm) for hm in heatmaps)
    del heatmaps
    return masked_heatmaps, gbs_heatmaps, img2txt_heatmaps


def get_model(params, image_model=None, caption_model=None):
    ckpt = torch.load(params.load_checkpoint_path)
    eval_module = tm.CondAlphaMapTM(up_depths=ckpt['decoder_depths'],
                                    backbone=ckpt['backbone'],
                                    img2txt=ckpt['img2txt'],
                                    )
    params.img2txt = ckpt['img2txt']

    # Load Checkpoint
    device = torch.device("cuda:{}".format(int(0)))

    eval_module.to(device)
    if image_model is None:
        eval_module, _, _, _ = utils.load_from_checkpoint(ckpt, eval_module, strict=True)
    else:
        eval_module.model.image_model = image_model
    if caption_model is not None:
        eval_module.model.caption_model = caption_model
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        logprint("Using {} GPUs!".format(torch.cuda.device_count()))
        eval_module.to(device)

    eval_module.eval()
    return eval_module


def evaluate(params, image_model=None, caption_model=None):
    evaluator = GBSEvaluator()
    utils.dry_run_decorator.set_dry_run(params.dry_run)
    utils.make_dirs(params, 'root_save_dir', ('save_output_dir',))
    norm_params = {'mean': [0.406, 0.456, 0.485], 'std': [0.225, 0.224, 0.229]}
    if params.verbosity > 0:
        logprint('Getting data loader')
    resize_transform = Resize(params.input_size, fill_value=np.nan, ar_perserve=False) \
        if params.batch_size > 1 else None
    other_transoforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**norm_params)]) \
        if params.batch_size > 1 else \
        transforms.Compose([transforms.ToPILImage(), transforms.Resize(params.input_size), transforms.ToTensor(),
                            transforms.Normalize(**norm_params)])

    data_loader, reader = get_bert_image_query_loader(params.data_root, params.split,
                                                      resize_transform,
                                                      other_transoforms,
                                                      params.batch_size, 0,
                                                      drop_last=False,
                                                      shuffle=False,
                                                      world_size=int(os.environ['WORLD_SIZE']),
                                                      rank=int(os.environ['RANK'])
                                                      )

    if params.verbosity > 0:
        logprint('Done getting data loader')
        logprint('Loading network')
    eval_module = get_model(params, image_model, caption_model)
    if params.verbosity > 0:
        logprint('Done loading network')

    try:
        dist_rank = int(os.environ.get('RANK', '0'))
        for i, eval_data in enumerate(tqdm(data_loader, disable=params.verbosity == 0)):

            with torch.no_grad():

                out_heatmaps, gbs_heatmaps, img2txt_heatmaps = forward_pass(eval_data, eval_module, params)
                annotations = [reader.get_annotations(ann_id, ims)[0] for ann_id, ims in zip(eval_data.ann_id,
                                                                                             eval_data.image_size)]

                if params.img2txt:
                    output_heatmap = [torch.sqrt(
                        torch.mul(sh, torch.nn.functional.interpolate(mgh, sh.shape[2:4])))
                                      for sh, mgh in zip(gbs_heatmaps, img2txt_heatmaps)]
                else:
                    output_heatmap = gbs_heatmaps

                evaluator.validate_batch(output_heatmap, annotations, params.arp_resize)
                if (not i % 10) and params.verbosity > 0:
                    evaluator.calc_results()
                    tqdm.write(f'Current Hit Acc step {i}/{len(data_loader)}: {evaluator.hit_acc}')

            del eval_data, out_heatmaps, gbs_heatmaps, img2txt_heatmaps
            if 0 < params.max_batches == i + 1:
                break
        if params.verbosity > 0:
            logprint(f'Done evaluation from rank {dist_rank}. Waiting for other ranks to finish....')
        device = torch.device("cuda:{}".format(0))
        if int(os.environ.get('WORLD_SIZE', 1)) > 1:
            dist.barrier()
            evaluator.gather_results(device)
        if params.verbosity > 0:
            logprint(f'Done collecting results from all nodes')
        evaluator.calc_results()
        out_file = os.path.join(params.save_output_dir, 'benchmark_results.txt') if not params.dry_run else None
        if params.verbosity > 0:
            evaluator.print_results(heatmap_naming='GbS', output_file=out_file)
            logprint(f'Finished Eval {params.training_experiment_path}')
        return evaluator.hit_acc

    except (TimeoutError, Exception, RuntimeError) as err:

        if params.debug_mode:
            raise err
        else:
            if not (err in (KeyboardInterrupt, InterruptedError)):
                track = traceback.format_exc()
                print(track)
            pass

        return np.nan, np.nan, np.nan, np.nan


def main():

    args = get_params()

    this_rank = args.local_rank
    all_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = all_gpus[this_rank]
    torch.backends.cudnn.benchmark = False
    dist_rank = int(os.environ.get('RANK', 0))
    hostname = os.environ['HOSTNAME']
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        logprint(f'Host:{hostname}, Rank: {dist_rank}'
                 f'Trying to init process group')
        torch.cuda.set_device(0)
        dist.init_process_group(backend='nccl', init_method='env://')
        logprint('Init process group')
    else:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
    print(f'debug {args.debug_mode}')
    if args.debug_mode:

        from cvar_pyutils.debugging_tools import set_remote_debugger
        set_remote_debugger(debug_port=12345)
    if not args.debug_mode:
        jobid = os.environ.get('LSB_JOBID', None)
        Logger.set_logger(os.path.join('.', "{}_{}.log".format(args.experiment_name.replace('/', '_'), jobid)))
        logprint('Workning on GPUS #{}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    if args.debug_mode:
        args.num_workers = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        logprint('Workning on {} GPUS # {}'.format(len(os.environ['CUDA_VISIBLE_DEVICES']),
                                                   os.environ['CUDA_VISIBLE_DEVICES']))
    evaluate(params=args)


def external(image_model, caption_model, args_list):
    args = Params().parse_args(args_list)
    return evaluate(args, image_model, caption_model)


if __name__ == '__main__':
    main()

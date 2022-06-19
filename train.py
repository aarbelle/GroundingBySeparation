from datetime import datetime
from data_loader import get_lmdb_image_pair_loader
import os
import time
import traceback
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from utils import logprint, Logger, GracefulKill, TerminateError, UnNormalize
from optimizers import get_optimizer
from parameters import get_params
import utils
import modules_train as tm
import torch.distributed as dist

utils.set_all_random_seeds(seed=0)

# noinspection PyTypeChecker


def train(params):

    local_rank = params.local_rank
    all_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = all_gpus[local_rank]
    dist_rank = int(os.environ.get('RANK', 0))
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        print(f'Process {dist_rank}:  Trying to init process group')
        print(f'Process {dist_rank}: MASTER_ADDR=', os.environ.get('MASTER_ADDR', 'no_addr'))
        print(f'Process {dist_rank}: MASTER_PORT=', os.environ.get('MASTER_PORT', 'no_port'))
        print(f'Process {dist_rank}: HOSTNAME=', os.environ.get('HOSTNAME', 'no_host'))
        print(f'Process {dist_rank}: WORLD_SIZE=', os.environ.get('WORLD_SIZE', 'no world'))
        print(f'Process {dist_rank}: RANK=', os.environ.get('RANK', 'no rank'))

        logprint('Trying to init process group')
        dist.init_process_group(backend='nccl', init_method='env://')
        logprint('Init process group')
    else:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
    utils.dry_run_decorator.set_dry_run(params.dry_run or dist_rank > 0)

    if params.save_code_snapshot and dist_rank == 0:
        now_str = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        utils.zip_code(os.path.dirname(os.path.abspath(__file__)), os.path.join(params.save_checkpoint_dir,
                                                                                f'backup_code_{now_str}.tar.gz'),)
    utils.make_dirs(params, 'root_save_dir', ('save_checkpoint_dir', 'save_log_dir', 'save_debug_dir'))
    norm_params = {'mean': [0.406, 0.456, 0.485], 'std': [0.225, 0.224, 0.229]}
    resize_aug = [transforms.ToPILImage()]

    ar = (3. / 4., 4. / 3.) if params.non_arp_resize else (1., 1.)
    resize_aug.append(transforms.RandomResizedCrop(params.input_size, scale=(0.7, 1.0), ratio=ar))
    resize = None
    val_path = list(params.lmdb_data_path).copy()
    for ind, vp in enumerate(val_path):
        if 'visual_genome' in vp:
            if vp.replace('visual_genome', 'coco') in val_path:
                val_path.pop(ind)
                continue
            val_path[ind] = vp.replace('visual_genome', 'coco')

    train_data_loader = get_lmdb_image_pair_loader(params.lmdb_data_path, 'train',
                                                   resize,
                                                   transforms.Compose(resize_aug + [
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ColorJitter(),
                                                            transforms.RandomGrayscale(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(**norm_params)
                                                        ], ),
                                                   params.batch_size, params.num_workers,
                                                   alpha_map=params.alpha_map,
                                                   drop_last=True,
                                                   caption_augs=params.caption_aug,
                                                   world_size=int(os.environ.get('WORLD_SIZE', 1)),
                                                   rank=dist_rank
                                                   )

    val_data_loader = get_lmdb_image_pair_loader(val_path, 'val',
                                                 resize,
                                                 transforms.Compose(resize_aug + [
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(**norm_params)
                                                      ], ),
                                                 params.batch_size * 2, params.num_workers // 4,
                                                 alpha_map=params.alpha_map,
                                                 drop_last=True,
                                                 shuffle=False,
                                                 seed=0,
                                                 world_size=int(os.environ.get('WORLD_SIZE', 1)),
                                                 rank=dist_rank
                                                 )
    unnorm = UnNormalize(**norm_params)

    train_module = tm.CondAlphaMapTM(decoder_depths=params.decoder_depths,
                                     backbone=params.backbone,
                                     img2txt_weight=params.img2txt_weight,
                                     )

    optimizer = get_optimizer(train_module.model, params)
    if dist_rank == 0:
        writers = utils.get_tboard_summary_writers(params.save_log_dir)
    else:
        writers = None
    writer_train, writer_val = writers if writers is not None else (None, None)
    # Load Checkpoint
    torch.cuda.set_device(0)
    device = torch.device("cuda:{}".format(0))
    train_module.to(device)

    if params.load_from_checkpoint:
        (train_module, optimizer, start_epoch, global_step) = utils.load_from_checkpoint(params.load_checkpoint_path,
                                                                                         train_module, optimizer,
                                                                                         strict=False)
        start_epoch += 1

    else:
        start_epoch = global_step = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.learning_rate_decay_step)

    if int(os.environ.get('WORLD_SIZE', 1)) > 1 and params.sync_bn:
        logprint('Using SyncBN')
        train_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(train_module)
        train_module.to(device)

    # noinspection PyArgumentList
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        train_module = nn.parallel.DistributedDataParallel(train_module, device_ids=[0], output_device=0,
                                                           find_unused_parameters=True)
    # noinspection PyTypeChecker
    tlen = len(train_data_loader)
    train_step = utils.TrainStep(tlen, global_step, start_epoch=start_epoch)

    if not params.debug_mode and not params.dry_run:
        # noinspection PyUnusedLocal
        def save_and_exit(ckpt_path, ts, train_m, opt):
            logprint('Got exit signal!')
            if dist_rank == 0:
                utils.save_checkpoint(ckpt_path, ts, train_m, opt, backbone=params.backbone,
                                      img2txt=params.img2txt_weight>0, decoder_depths=params.decoder_depths)
            exit(0)

        # noinspection PyUnusedLocal
        gk = GracefulKill(save_and_exit, (params.save_checkpoint_path, train_step, train_module, optimizer))
        logprint('GracefulKill was set.')

    def forward_pass(data):
        # prepare inputs:
        if torch.cuda.device_count() == 1:
            data = data.to(device)
        mask = data.mask1 * data.mask2

        blended_image = (data.image1 * data.alpha + data.image2 * (1 - data.alpha)) * mask

        inputs = (blended_image, data.caption_emb1, data.caption_emb2, data.image1, data.image2,
                  data.caption_mask1, data.caption_mask2, mask[:, :1], data.alpha,  data.alpha1, data.alpha2,
                  data.captions1, data.captions2)

        loss = train_module(inputs, use_neg=params.neg_weight,
                            run_no_cond=params.adv_weight)
        if hasattr(train_module, 'module'):
            vis_scalars_dict, vis_images_dict = train_module.module.vis_scalars, train_module.module.vis_images
        elif hasattr(train_module, 'vis_scalars'):
            vis_scalars_dict, vis_images_dict = train_module.vis_scalars, train_module.vis_images
        else:
            vis_scalars_dict = vis_images_dict = {}
        if len(vis_scalars_dict) == 0:
            raise ValueError('Missing Visualizations')
        if loss is not None:
            loss = tuple((loss_.mean(dim=0) for loss_ in loss)) if isinstance(loss, tuple) else loss.mean(dim=0)

        return loss, vis_scalars_dict, vis_images_dict

    try:
        best_val_loss = None

        running_loss = torch.tensor(0.0)
        torch.autograd.set_detect_anomaly(True)

        running_count = 0
        for epoch_num in range(start_epoch, params.num_epochs):  # loop over the dataset multiple times

            i = -1
            if int(os.environ.get('WORLD_SIZE', 1)) > 1:
                torch.distributed.barrier()

            # noinspection PyUnresolvedReferences
            train_data_loader.sampler.set_epoch(epoch_num)
            for train_data in train_data_loader:
                if int(os.environ.get('WORLD_SIZE', 1)) > 1:
                    torch.distributed.barrier()
                i += 1

                loss_tensor, vis_scalars, vis_images = forward_pass(train_data)
                optimizer.zero_grad()
                # Backward pass
                loss_tensor.backward(retain_graph=False)
                # gradient step
                optimizer.step()
                train_step.step()
                # save statistics
                this_loss = loss_tensor.detach().item()
                running_loss += this_loss
                running_count += 1

                scheduler.step()

                if not train_step.global_step % params.log_train_interval and train_step.global_step > 0:
                    running_loss = running_loss / running_count
                    print_str = 'train: [epoch {:d}, step {:5d}, global step {:d}] ' \
                                'loss: {:0.3f}'.format(train_step.epoch, i + 1,
                                                       train_step.global_step,
                                                       running_loss, )
                    # save tensorboard
                    tb_vis_scalars, tb_vis_images = utils.prepare_for_tboard(vis_scalars, vis_images, unnorm)
                    scalars_dict = {'loss': running_loss,
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'weight decay': params.weight_decay, }

                    tb_vis_scalars.update(scalars_dict)
                    if dist_rank == 0:
                        utils.write_to_tensorboard(writer_train, train_step.global_step, tb_vis_scalars, tb_vis_images)

                    running_loss = 0.0  # initialize statistics
                    running_count = 0

                    logprint(print_str)

                    del tb_vis_images, tb_vis_scalars, scalars_dict

                del train_data, loss_tensor, vis_scalars, vis_images
                # Validation
                # noinspection PyTypeChecker
                if int(os.environ.get('WORLD_SIZE', 1)) > 1:
                    torch.distributed.barrier()
                if (train_step.global_step > 0 and not train_step.global_step % params.validation_interval or
                        i == len(train_data_loader) - 1):
                    logprint('Starting validation')
                    running_val_loss = 0.0
                    val_vis_scalars = {}
                    val_vis_images = {}
                    vi = 0
                    v_step = -1
                    start_val_time = time.time()
                    for val_data in val_data_loader:

                        with torch.no_grad():
                            v_step += 1
                            loss_tensor, _, _ = forward_pass(val_data)
                            if v_step == 0:
                                loss_tensor, val_vis_scalars, val_vis_images = forward_pass(val_data)
                            else:
                                loss_tensor, _, _ = forward_pass(val_data)

                            running_val_loss += loss_tensor.detach().item()
                            vi += 1
                            # noinspection PyTypeChecker
                            if (time.time() - start_val_time) / 60 > params.max_val_time:
                                logprint(f'Validation step {v_step}/{len(val_data_loader)}. Time exceeded')
                                break
                            if v_step + 1 < len(val_data_loader):
                                del loss_tensor

                    train_module.train()
                    running_val_loss = running_val_loss / vi
                    print_str = 'validation: [epoch {:d}, step {:5d}, global step {:d}]  ' \
                                'loss: {:0.3f}'.format(train_step.epoch, i + 1,
                                                       train_step.global_step,
                                                       running_val_loss)

                    val_vis_scalars, val_vis_images = utils.prepare_for_tboard(val_vis_scalars, val_vis_images,
                                                                               unnorm)

                    val_vis_scalars['loss'] = running_val_loss
                    logprint(print_str)
                    if dist_rank == 0:
                        utils.write_to_tensorboard(writer_val, train_step.global_step, val_vis_scalars, val_vis_images)
                    if best_val_loss is None or running_val_loss < best_val_loss:
                        best_val_loss = running_val_loss
                        utils.save_checkpoint(params.save_checkpoint_path + '.best_val', train_step, train_module,
                                              optimizer, backbone=params.backbone,img2txt=params.img2txt_weight>0,
                                              decoder_depths=params.decoder_depths)
                    del loss_tensor, val_vis_scalars, val_vis_images, val_data

                if params.max_steps and train_step.global_step >= params.max_steps:
                    break  # Exit inner loop when max steps is reached

            # Save checkpoint at end of epoch
            if dist_rank == 0:
                utils.save_checkpoint(params.save_checkpoint_path, train_step, train_module, optimizer,
                                      backbone=params.backbone, img2txt=params.img2txt_weight>0,
                                      decoder_depths=params.decoder_depths
                                      )
            if params.max_steps and train_step.global_step >= params.max_steps:
                break  # Exit outer loop when max steps is reached
        logprint('Finished Training')

    except (TimeoutError, Exception, RuntimeError, TerminateError) as err:
        if dist_rank == 0:
            utils.save_checkpoint(params.save_checkpoint_path, train_step, train_module, optimizer,
                                  backbone=params.backbone, img2txt=params.img2txt_weight>0,
                                  decoder_depths=params.decoder_depths)

        if params.debug_mode:
            print('In debug mode')
            raise err
        else:
            if not (err in (KeyboardInterrupt, InterruptedError)):
                track = traceback.format_exc()
                print(track)
            pass

        if not params.dry_run and dist_rank == 0:
            # noinspection PyUnresolvedReferences
            writer_train.close()
            # noinspection PyUnresolvedReferences
            writer_val.close()


def main():
    args = get_params()
    if not args.debug_mode:
        Logger.set_logger(os.path.join('.', "{}_{}_{}.log".format(args.experiment_name.replace('/', '_'),
                                                                  os.environ.get('HOSTNAME', 'localhost'),
                                                                  args.local_rank)))
    else:
        args.num_workers = 0
    logprint('Working on GPUS: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    train(params=args)


if __name__ == '__main__':
    main()

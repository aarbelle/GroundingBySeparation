import cv2
import glob
import tarfile
import textwrap
import torch
import typing
import os
import shutil
import random
import sys
import time
import signal
import threading
import numpy as np
import torch.optim as optim
# noinspection PyPackageRequirements
from tap import Tap
from tensorboardX import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Tuple, Union
from multiprocessing import current_process


SIG_NUMS = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 25)


class TerminateError(Exception):
    pass


class GracefulKill(object):
    err = TerminateError

    def __init__(self, cleanup_fun: typing.Callable = None, cleanup_args: typing.Iterable = None,
                 cleanup_kwargs: dict = None, sig_nums: typing.Iterable = SIG_NUMS):
        self._cleanup_fun = None
        self._cleanup_args = None
        self._cleanup_kwargs = None
        self.set_cleanup_function(cleanup_fun, cleanup_args, cleanup_kwargs, sig_nums)
        self._caught_signal = False

    def set_cleanup_function(self, cleanup_fun: typing.Callable = None, cleanup_args: typing.Iterable = None,
                             cleanup_kwargs: dict = None, sig_nums: typing.Iterable = SIG_NUMS):
        self._cleanup_fun = cleanup_fun
        self._cleanup_args = cleanup_args if cleanup_args is not None else tuple()
        self._cleanup_kwargs = cleanup_kwargs if cleanup_kwargs is not None else {}
        for signum in sig_nums:

            try:
                signal.signal(signum, self.exit_fn)
            except (OSError, RuntimeError, ValueError, Exception):
                print(f'Could not set {cleanup_fun} for kill signal {signum}')
                pass

    def exit_fn(self, sig_num, _):
        if self._caught_signal:
            return
        if current_process().name == 'MainProcess':
            self._caught_signal = sig_num
            if threading.current_thread() is threading.main_thread():
                logprint(f'Caught signal {sig_num} in main process, main thread')
            else:
                time.sleep(30)
                return
        else:
            time.sleep(30)
            return

        logprint(f'Got kill signal {sig_num}. Need to quit process.')
        if self._cleanup_fun is None:
            logprint('No clean up function was set. Quitting process')
            sys.exit()
        else:
            logprint(f'Running {self._cleanup_fun} with arguments {len(self._cleanup_args)+len(self._cleanup_kwargs)}.')
            self._cleanup_fun(*self._cleanup_args, **self._cleanup_kwargs)
            # logprint('Exiting process.')
            # sys.exit()


class UnNormalize(object):
    """
    The opposite of torch.transform.Normalize
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape((-1, 1, 1))
        self.std = torch.tensor(std).reshape((-1, 1, 1))

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be un-normalized.
        Returns:
            Tensor: Un-Normalized image.
        """
        if tensor.is_cuda and not self.mean.is_cuda:
            self.mean = self.mean.to('cuda')
            self.std = self.std.to('cuda')
        elif not tensor.is_cuda and self.mean.is_cuda:
            self.mean = self.mean.to('cpu')
            self.std = self.std.to('cpu')

        return (tensor * self.std) + self.mean


class TimeRecorder(object):

    def __init__(self, sort_by_percentage: bool = True, time_format: str = '0.3e'):
        """

        Args:
            sort_by_percentage: When printing, sort the elements by percentage of total time in decreasing order
            time_format: Formatting type for time when printing
        """
        self.sort_by_percentage = sort_by_percentage
        self.time_format = time_format
        self.timings = []
        self.comments = []

    def reset(self):
        self.timings = []
        self.comments = []

    def tic(self, comment: str = None) -> Tuple[float, str]:
        """

        Args:
            comment: Comment to log on start of timing
        """
        self.reset()
        return self(comment)

    def toc(self, comment: str = None) -> Tuple[float, str]:
        """

        Args:
            comment: Comment to log on timing step
        """
        return self(comment)

    def __call__(self, comment: str = None) -> Tuple[float, str]:
        self.timings.append(time.time())
        if comment is None:
            comment = 'Recorded Time {}'.format(len(self.timings) - 1)
        self.comments.append(comment)
        return self.timings[-1], comment

    def print(self, sort_by_percentage: bool = None, time_format: str = None):
        """

        Args:
            sort_by_percentage: When printing, sort the elements by percentage of total time in decreasing order
                                overrides defaults provided at init
            time_format: Formatting type for time when printing
                         overrides defaults provided at init

        """
        if sort_by_percentage is None:
            sort_by_percentage = self.sort_by_percentage
        if time_format is None:
            time_format = self.time_format

        if len(self.timings) < 2:
            logprint('Not enough records to print')
            return
        time_array = np.array(self.timings)
        total_time = time_array[-1] - time_array[0]
        interval_time = time_array[1:] - time_array[:-1]
        interval_prints = [f'"{l1}" to "{l2}": {t:{time_format}} ({t / total_time * 100:0.2f}%)' for l1, l2, t in
                           zip(self.comments[:-1], self.comments[1:], interval_time)]
        logprint(f'Total Time: {total_time:{time_format}}')
        if sort_by_percentage:
            interval_prints = [p for _, p in sorted(zip(interval_time, interval_prints), key=lambda pair: pair[0])]
            interval_prints.reverse()
        for p in interval_prints:
            logprint(p)
        return

    def __str__(self):
        sort_by_percentage = self.sort_by_percentage
        time_format = self.time_format

        if len(self.timings) < 2:
            logprint('Not enough records to print')
            return
        time_array = np.array(self.timings)
        total_time = time_array[-1] - time_array[0]
        interval_time = time_array[1:] - time_array[:-1]
        interval_prints = [f'"{l1}" to "{l2}": {t:{time_format}} ({t / total_time * 100:0.2f}%)' for l1, l2, t in
                           zip(self.comments[:-1], self.comments[1:], interval_time)]
        logprint(f'Total Time: {total_time:{time_format}}')
        if sort_by_percentage:
            interval_prints = [p for _, p in sorted(zip(interval_time, interval_prints), key=lambda pair: pair[0])]
            interval_prints.reverse()

        return '\n'.join(interval_prints)

    def __repr__(self):
        return "{:s}({!r})".format(self.__class__, self.__dict__)


class TrainStep(object):

    def __init__(self, epoch_size: int, start_global_step: int = 0, start_epoch: int = 0):
        self.epoch_size = epoch_size
        self.global_step = start_global_step
        self.epoch = self.global_step // self.epoch_size
        self.epoch_step = self.global_step % self.epoch_size
        self.start_epoch = start_epoch
        self.local_step = 0

    def step(self):
        self.global_step += 1
        self.local_step += 1
        self.epoch = self.start_epoch + self.local_step // self.epoch_size
        self.epoch_step = self.local_step % self.epoch_size

    def __repr__(self):
        return str(self.__dict__)


class DryRunDecorator(object):
    def __init__(self, dry_run=False):
        self._dry_run = dry_run

    def __call__(self, fun):

        def wrapper(*args, **kwargs):

            if self._dry_run:
                logprint(f'Warning dry_run flag is on! skipping function {fun.__name__}')
                return None
            else:
                outputs = fun(*args, **kwargs)
                return outputs

        return wrapper

    def set_dry_run(self, dry_run=True):
        self._dry_run = dry_run


dry_run_decorator = DryRunDecorator()


# noinspection PyUnresolvedReferences
@dry_run_decorator
def make_dirs(params: Tap, root_dir: str, sub_dirs: typing.Tuple[str, ...] = None):
    root_dir = getattr(params, root_dir)
    all_dirs = [getattr(params, k) for k in sub_dirs]
    all_dirs.append(root_dir)
    os.umask(0)

    if os.path.exists(root_dir):
        backup_num = ''
        if params.override_experiment_with_backup:
            while os.path.exists(root_dir + '_bkp{}'.format(backup_num)):
                backup_num = 1 if backup_num == '' else backup_num + 1
            shutil.move(root_dir, root_dir + '_bkp{}'.format(backup_num))
        elif params.override_experiment:
            shutil.rmtree(root_dir)
    for d in all_dirs:
        os.makedirs(d, exist_ok=True, mode=0o777)
    now_str = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    params.save(os.path.join(root_dir, 'params_{}.json'.format(now_str)))
    params.save(os.path.join(root_dir, 'params.json'))


def get_save_model(module: torch.nn.Module):
    if (torch.cuda.device_count() >= 1 or int(os.environ.get('WORLD_SIZE', 1))>1) and hasattr(module, 'module'):
        module = module.module
    return module


# noinspection PyUnresolvedReferences
def load_from_checkpoint(load_checkpoint_path: typing.Union[str, dict], module: torch.nn.Module,
                         optimizer: optim.Optimizer = None, strict=True):
    start_epoch = global_step = 0
    if isinstance(load_checkpoint_path, dict) or os.path.exists(load_checkpoint_path):
        try:
            if isinstance(load_checkpoint_path, str):
                logprint(f'Loading checkpoint from file: {load_checkpoint_path}')
                checkpoint = torch.load(load_checkpoint_path)
            else:
                checkpoint = load_checkpoint_path
            state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                if k.startswith('model.image_model.up_blocks'):
                    k = k.replace('model.image_model.up_blocks', 'model.image_model.decoder_blocks')
                if k=='model.image_model.scale_conv_up.weight':
                    continue
                state_dict[k]=v
            # state_dict = OrderedDict([(k.replace('model.image_model.up_blocks', 'model.image_model.decoder_blocks'), v)
            #                           if k.startswith('model.image_model.up_blocks') else (k, v)
            #                           for k, v in checkpoint['model_state_dict'].items()])
            module.load_state_dict(state_dict, strict=strict)
            if optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except ValueError:
                    logprint('Could not reload optimizer from checkpoint. Resuming with randomly initialized optimizer')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint.get('global_step', -1)

        except (EOFError, RuntimeError) as err:
            if isinstance(load_checkpoint_path, str):
                logprint(f'Could not load checkpoint: {load_checkpoint_path} \nUsing random weights.')

            else:
                logprint(f'Could not load checkpoint. \nUsing random weights.')
            raise err
    else:
        logprint('Using random weights.')
    return module, optimizer, start_epoch, global_step


# noinspection PyUnresolvedReferences
@dry_run_decorator
def save_checkpoint(save_checkpoint_path: str, ts: TrainStep, module: torch.nn.Module,
                    optimizer: optim.Optimizer, **kwargs):
    save_model = get_save_model(module)
    save_dict = {
        'epoch': ts.epoch,
        'global_step': ts.global_step,
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_class': save_model.__class__.__name__,
        }
    save_dict.update(kwargs)
    torch.save(save_dict, save_checkpoint_path)
    logprint('Saved checkpoint to {}'.format(save_checkpoint_path))
    os.chmod(save_checkpoint_path, 0o777)


@dry_run_decorator
def write_to_tensorboard(writer: SummaryWriter, step: int, scalar_dict: dict = None, image_dict: dict = None,
                         hist_dict: dict = None):
    if writer is None:
        logprint('Got writer None. Skipping write_to_tensorboard')
        return
    if scalar_dict is not None:
        for tag, val in scalar_dict.items():
            try:
                writer.add_scalar(tag, val, step)
            except ValueError:
                logprint(f'Could not add {tag} to tensorboard')
    if image_dict is not None:
        for tag, val in image_dict.items():
            try:
                writer.add_image(tag, val, step)
            except ValueError:
                logprint(f'Could not add {tag} to tensorboard')
    if hist_dict is not None:
        for tag, val in hist_dict.items():
            if val is np.nan:
                continue
            try:
                writer.add_histogram(tag, val, step)
            except ValueError:
                logprint(f'Could not add {tag} to tensorboard')


def prepare_for_tboard(vis_scalars_dict: dict, vis_images_dict: dict, unnorm, make_grid=True):
    for k, v in vis_scalars_dict.items():
        if isinstance(v, torch.Tensor):
            vis_scalars_dict[k] = v.mean().item()

    grid_rows = []
    max_x = 0
    for k, v in vis_images_dict.items():
        # v: (B,C,H,W)
        if isinstance(v, tuple) or isinstance(v, list):
            try:
                tag = text2img(k, v[0].shape[2:4])

            except IndexError:
                tag = np.zeros_like(v[0][0])
                logprint(f'Caught IndexError. v.shape: {v.shape}, tag: {k}')

            tag = torch.tensor(tag, dtype=torch.float, device='cpu').permute(2, 0, 1)

            v = torch.cat(v, 3)  # stack on x axis
        else:
            try:
                tag = text2img(k, v.shape[2:4])
            except IndexError:
                logprint(f'Caught IndexError. v.shape: {v.shape}, tag: {k}')
                tag = np.zeros_like(v[0])
            tag = torch.tensor(tag, dtype=torch.float, device='cpu').permute(2, 0, 1)

        if v.shape[1] == 3:
            v = unnorm(v)  # unnormalise
        elif make_grid:
            v = torch.cat([v] * 3, dim=1)  # convert to RGB for grid v: (3,H,W)
        # v: (B,3,H,W)
        v = v.detach().to('cpu')  # move to cpu
        v = v[0]  # extract one image  v: (C,H,W)
        v = v.flip(0)  # BGR2RGB

        v = v.numpy()  # convert to numpy
        v = v.clip(0, 1)  # clip to 0,1
        if make_grid:

            v = np.concatenate((tag.numpy(), v), 2)
            grid_rows.append(v)
            max_x = np.maximum(max_x, v.shape[2])
        vis_images_dict[k] = v
    if make_grid:
        for rind, gr in enumerate(grid_rows):
            sz, sy, sx = gr.shape
            if sx < max_x:
                z = np.zeros((sz, sy, max_x - sx))
                grid_rows[rind] = np.concatenate((gr, z), axis=2)
        grid_img = np.concatenate(grid_rows, 1)
        vis_images_dict = {'Grid': grid_img}
        # vis_images_dict['Grid'] = grid_img

    return vis_scalars_dict, vis_images_dict


@dry_run_decorator
def get_tboard_summary_writers(save_log_dir: str) -> typing.Tuple[SummaryWriter, SummaryWriter]:
    writer_train = SummaryWriter(os.path.join(save_log_dir, 'train'))
    writer_val = SummaryWriter(os.path.join(save_log_dir, 'val'))
    return writer_train, writer_val


def get_heatmap_max_coord(heatmap: np.ndarray):
    ind = heatmap.argmax()
    x = ind % heatmap.shape[1]
    y = ind // heatmap.shape[1]
    return y, x

# noinspection PyUnresolvedReferences
def set_all_random_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


@dry_run_decorator
def zip_code(root_dir, out_filename, max_file_size=1e6, use_ext=('*.py', '*.yml')):
    files = []
    for ext in use_ext:
        files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    os.umask(0)
    os.makedirs(os.path.dirname(out_filename), exist_ok=True, mode=0o777)
    tar_fp = tarfile.open(out_filename, 'w:gz')
    for f in files:
        try:
            if os.path.getsize(f) < max_file_size:
                arcname = f.replace(f'{os.path.dirname(root_dir)}', '.', 1)
                tar_fp.add(f, arcname=arcname)
        except OSError:
            pass
    tar_fp.close()
    logprint(f'Saved backup code to file: {out_filename}')
    

def text2img(text, shape=None):

    if isinstance(text, np.ndarray):
        try:
            text = ' '.join(t.decode('UTF-8') for t in text)
        except:
            text = ' '.join(t for t in text)
    font_size = 2
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, font_size, font_thickness)[0]
    num_splits = int(np.ceil(np.sqrt(textsize[0]/(textsize[1]*4))))
    line_width = int(textsize[0]//num_splits)
    wrapped_text = textwrap.wrap(text, width=len(text)//num_splits)
    boundary = max(80, 5*len(wrapped_text)+20)
    img = np.zeros((line_width + boundary, line_width+boundary, 3), dtype='uint8')
    height, width, channel = img.shape

    text_img = np.ones((height, width))
    # print(text_img.shape)
    y0 = textsize[1] + 5
    for i, line in enumerate(wrapped_text):

        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        gap = textsize[1] + 5
        y = 10 + i * gap + y0
        x = 10  # for center alignment => int((img.shape[1] - textsize[0]) / 2)
        cv2.putText(img, line, (x, y), font,
                    font_size,
                    (255, 255, 255),
                    font_thickness,
                    lineType=cv2.LINE_AA)

    if shape is not None:
        img = cv2.resize(img, (shape[1], shape[0]))
    return img


def gray2rgb(img, cmap='viridis'):
    cmap = plt.get_cmap(cmap)
    if img.shape[2] == 1:
        rgba = cmap(img[..., 0])
        img = np.delete(rgba, 3, 2)
    return img


def convert_img_tensor2np(img: torch.Tensor):
    img = np.flip(img.permute(1, 2, 0).cpu().numpy(), 2)
    return img


class Logger(object):
    def __init__(self, log_path='log.txt'):
        """
        To set the Logger add the following lines to your code:

        >> Logger.set_logger(PATH_TO_LOG_FILE)

        or:

        >> import sys
        >> logger=Logger(PATH_TO_LOG_FILE)
        >> sys.stdout = logger
        >> sys.stderr = logger

        """
        self.terminal = sys.stdout
        self.log = open(log_path, "a", 1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    @classmethod
    def set_logger(cls, log_path='log.txt'):
        logger = cls(log_path)
        sys.stderr = logger
        sys.stdout = logger


def logprint(*args, **kwargs):
    flush = kwargs.get('flush', None)
    kwargs['flush'] = flush if flush is not None else True
    print('[{}]'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), *args, **kwargs)



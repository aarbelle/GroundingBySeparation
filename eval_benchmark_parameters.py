# noinspection PyPackageRequirements
from tap import Tap
import os
from parameters import Params as TrainParams

LMDB_DATA_ROOT = os.environ.get('LMDB_DATA_ROOT', './data')


# noinspection LongLine
class ParamsBase(Tap):
    """
    All default parameters should be set here
    """

    """------General------"""
    experiment_name: str = 'evaluate_benchmark'  # Name of experiment, used to create subdirectory for logs and checkpoints
    training_experiment_path: str = None
    load_checkpoint_path: str = None  # Path to a specific checkpoint, if not given, loads the best_val
    config_file: str = None  # path to configuration file
    dry_run: bool = False  # Does not save any outputs - For debug purpose only.
    override_experiment: bool = False  # Use with caution! Deletes all logs and checkpoint of <experiment_name> if exists.
    override_experiment_with_backup: bool = False  # Moves logs and checkpoint of <experiment_name> to <experiment_name>_bkp if exists.
    max_batches: int = 0  # Number of batches to evaluate. set 0 to evaluate full dataset

    """------Data reader------"""
    data_root: str = os.path.join(LMDB_DATA_ROOT, 'flickr30k')  # Root directory for benchmark data
    split: str = 'test'  # split, either 'train', 'val' or 'test'
    batch_size: int = 1  # evaluation batch size
    num_workers: int = 3  # Number of workers for data loader
    input_size: int = 512  # Input image resize.
    """------Debugging--------"""
    debug_mode: bool = False
    verbosity: int = 1  # verbosity level. set [0-1] to quit outputs
    local_rank: int = 0  # set by pytorch ddp module
    """
    END DEFAULT PARAMETERS
    """

    # noinspection PyAttributeOutsideInit
    def process_args(self):
        if self.experiment_name is None or type(self) is ParamsBase:
            return
        if self.training_experiment_path is None:
            raise ValueError('parameter --training_experiment_path is required!')
        if not os.path.exists(self.training_experiment_path):
            raise FileNotFoundError(f'Cannot load experiment {self.training_experiment_path}')
        self.save_output_dir = os.path.join(self.training_experiment_path, self.experiment_name)
        self.root_save_dir = self.save_output_dir
        if self.load_checkpoint_path is None:
            self.load_checkpoint_path = os.path.join(self.training_experiment_path, 'checkpoints', 'checkpoint.dict')
        self.arp_resize = False  # not train_params.non_arp_resize

    def add_arguments(self) -> None:
        self.add_argument('-e', '--experiment_name')


class Params(ParamsBase):

    def add_arguments(self) -> None:
        config_file = ParamsBase(description=self.description).parse_args(known_only=True).config_file
        if config_file is not None:
            self.load(config_file)
        super().add_arguments()

    def _log_all(self):  # override tap git usage
        return self.as_dict()


def get_params():
    args = Params(description='Evaluate Benchmark Grounding by Separation').parse_args()
    args.process_args()
    return args


if __name__ == '__main__':
    pass

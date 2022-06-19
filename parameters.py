# noinspection PyPackageRequirements
from tap import Tap
from typing import List
import os

LMDB_DATA_ROOT = os.environ.get('LMDB_DATA_ROOT', './data')


# noinspection LongLine


class ParamsBase(Tap):
    """
    All default parameters should be set here
    """

    """------General------"""
    experiment_name: str = None  # Name of experiment, used to create subdirectory for logs and checkpoints
    config_file: str = None  # path to configuration file
    dry_run: bool = False  # Do not save any outputs - For debug purpose only.
    override_experiment: bool = False  # Use with caution! Deletes all logs and checkpoint of <experiment_name> if exists.
    override_experiment_with_backup: bool = False  # Moves logs and checkpoint of <experiment_name> to <experiment_name>_bkp if exists.

    """-------Alpha Config"""
    alpha_map: List[float] = [1., 0.5, 2, 0.5] # A list of pairs of numbers, the firs number indicates the alpha type (1-perln,2-gaussian) The second is the percentage of the batch [0-1]

    """------Network Config------"""
    backbone: str = 'vgg'  # either 'vgg' or 'pnasnet5large'
    decoder_depths: List[int] = (512,)  # List of depths for decoder layers.
    load_from_checkpoint: bool = True  # A flag to to load from checkpoint or not
    load_checkpoint_path: str = None  # Path to checkpoint, if not given looks for latest checkpoint in directory if exists.
    """------Data reader------"""
    input_size: int = 512  # size of input image, the image will be resiezed to a square while perserving aspect ratio
    lmdb_data_path: List[str] = (os.path.join(LMDB_DATA_ROOT, 'coco'),) # Path to trainind data
    output_dir: str = os.path.join('.', 'Outputs')  # Path to save output files
    batch_size: int = 3  # Training batch size
    num_workers: int = 2  # Number of workers for data loader
    non_arp_resize: bool = False  # perform non aspect perserving resize
    caption_aug: List[str] = 'drop:0.5'  # caption augmentaion can be drop:0.5 for 50% drop rate
    """------Optimizer and Loss------"""
    neg_weight: float = 1  # weight for negative loss
    adv_weight: float = 1  # weight for adversarial loss
    img2txt_weight: float = 0.1 # weight for image2text loss
    optimizer: str = 'adam'  # Training optimizer, currently implemented 'sgd' or 'adam'
    learning_rate: float = 0.0001  # Training learning rate
    backbone_lr: float = 0.0001  # Training learning rate for backbone
    momentum: float = 0.9  # Training momentum for SGD optimizer
    beta: List[float] = [0.9, 0.999]  # Training momentum for Adam optimizer
    weight_decay: float = 0.000000001  # Training weight decay regularization
    learning_rate_decay_step: int = 50000  # Step number for lr scheduler decay

    """------Training and Logging------"""
    num_epochs: int = 10  # Number of epochs in  training session
    log_train_interval: int = 1000  # Number of iterations between each print of train loss and accuracy
    validation_interval: int = 5000  # Number of iterations between each validation step
    max_val_time: float = 10  # maximum time (in minutes) to spend on validation. breaks validation loop at this time
    max_steps: int = None  # Maximum number of steps, used mainly for debugging and unit tests

    """------Distributed Training------"""
    sync_bn: bool = False  # use sync batch norm for DDP
    local_rank: int = 0  # set by torch multiprocessing
    """------Debugging--------"""
    debug_mode: bool = False  # Used for debug purposes
    save_code_snapshot = False  # Save a zip of the code when running, used for debug/reproducability
    """ 
    END DEFAULT PARAMETERS
    """

    # noinspection PyAttributeOutsideInit

    def add_arguments(self) -> None:
        self.add_argument('-e', '--experiment_name')
        pass


class Params(ParamsBase):

    def add_arguments(self) -> None:
        config_file = ParamsBase(description=self.description).parse_args(known_only=True).config_file
        if config_file is not None:
            self.load(config_file)
        super().add_arguments()

    # noinspection PyAttributeOutsideInit
    def process_args(self):
        if not self._parsed:
            return

        # process parameters for saveing
        self.root_save_dir = os.path.join(self.output_dir, self.experiment_name)
        self.save_checkpoint_dir = os.path.join(self.root_save_dir, 'checkpoints')
        self.save_log_dir = os.path.join(self.root_save_dir, 'logs')
        self.save_debug_dir = os.path.join(self.root_save_dir, 'debug')
        self.save_checkpoint_path = os.path.join(self.save_checkpoint_dir, 'checkpoint.dict')
        if self.load_from_checkpoint is True and self.load_checkpoint_path is None:
            self.load_checkpoint_path = os.path.join(self.save_checkpoint_dir, 'checkpoint.dict')

    def _log_all(self):  # override tap git usage
        return self.as_dict()


def get_params():
    args = Params(description='Train Grounding by Separation').parse_args()
    args.process_args()
    return args

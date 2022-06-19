# GroundingBySeparation
Visual Image Grounding by image separation.

# Installation:
## Requirements
1. Linux machine
1. At least one NVIDIA GPU
1. At least CUDA 10.2
1. Anaconda (Installation instructions: https://docs.anaconda.com/anaconda/install/)
## Install Dependencies
Clone the repository:
`git clone TBD`
Enter the directory:
`cd GroundingBySeparation`
Create and activate the conda environment:
```shell script
conda deactivate # deactivate any active environments
conda create -n gbs python=3.6 # install the conda environment with conda dependencies
conda activate gbs # activate the environment
conda install -c conda-forge libjpeg-turbo
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt # install pip requirements
```

# Data Preperations
We follow the data preperations of Akbari et al. 2019: 
https://github.com/hassanhub/MultiGrounding

Please prepare the data in the same format as described in the above repository.

After data preperation, either place the data in `<GBS_ROOT_DIR>/data` or  set the path in the environment variable `LMDB_DATA_ROOT`

# Training and Evaluation

## Run the training script
To train a network run the following command from the project root directory:
```shell script
python3 train.py --experiment_name my_first_run
```
#### Note: 
The parameter `--experiment_name` sets the sub-directory to save all the model checkpoints and tensorboard logs.

This name needs to be unique for each experiment. If the same name is used, the training will automatically continue 
from the last saved checkpoint (unless define otherwise in the parameters) 
### Optional Parameters
A list of all the train parameters and the discription can be seen in 
```shell script
python3 train.py --help
```


## Run the evaluation script
To evaluate the test results for the experiment `my_first_run` run the following command:
```shell script
python3 evaluate_benchmark.py --experiment_name evaluate_my_first_run --training_experiment_path ./Outputs/my_first_run
```

### Optional Parameters
A list of all the train parameters and the discription can be seen in 
```shell script
python3 evaluate_benchmark.py --help
```


## DDP Support
Our code fully supports Distributed Data Parallel (DDP) and can work on multiple gpus and multiple nodes.
In order to run the code with more than one GPU (or node) just add the following prefix to the `train.py` or 
`evaluate_benchmark.py` calls.
```shell script
python3  -m torch.distributed.launch --nproc_per_node=<NUM_GPUS_PER_NODE> --nnodes=<TOTAL_NUMBER_OF_NODES> --node_rank=<CURRENT_NODE_RANK> --master_addr=<HOSTNAME_OF_MASTER_NODE> --master_port=<SOME_PORT> train.py --args
```
for example to run on a single node with 8 gpus use the following:

```shell script
python3  -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=54321 train.py --args
``` 
to run on two nodes (with hostnames `node1` and `node2`) with four gpus each run the following:
On `node1`:
```shell script
python3  -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=node1 --master_port=54321 train.py --args
``` 
On `node2`:
```shell script
python3  -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=node1 --master_port=54321 train.py --args
``` 


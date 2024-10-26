import torch
from datetime import datetime
from typing import List, Tuple, Dict, Any
import os
from Models import *

Tensor = torch.Tensor
Module = torch.nn.Module
FP32 = torch.float32

# diffusion steps
T = 500

# padded trajectory length (maximal length)
TRAJ_LEN = 512

### Training Control -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
init_lr = 1e-4
lr_reduce_factor = 0.5
lr_reduce_patience = 50
batch_size = 50
epochs = 1000
log_interval = 10
mov_avg_interval = 15 * T

### Dataset Configs -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
dataset_name = "apartments"
# dataset_name = "Xian"
# dataset_name = "Chengdu"

### Diffusion Configs -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
diffusion_args = {
    "min_beta": 0.0001,
    "max_beta": 0.05,
    "max_diffusion_step": T,
    "scale_mode": "quadratic",
    "skip_step": 20,
}

### Diffusion Configs -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

model_name = "TW_MultiSeq_Add"
# model_name = "TW_MultiSeq_Cat"
# model_name = "TW_MultiSeq_CA"
# model_name = "TW_SingleSeq"
# model_name = "TW_MultiVec"

embed_dim = 6

### Advanced settings -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

log_dir = f"./Runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
save_dir = log_dir

if dataset_name == "apartments":
    from Dataset.DatasetApartments import DatasetApartments
    from BatchManagers import ApartmentsBatchManager as BatchManager
    dataset_args = {
        "max_len": TRAJ_LEN,
        "load_path": "./Dataset/dataset.pth",
    }
else:
    from Dataset.DatasetTaxi import DatasetTaxi
    from BatchManagers import TaxiBatchManager as BatchManager
    dataset_args = {
        "max_len": TRAJ_LEN,
        "load_path": f"./Dataset/{dataset_name}_nov_cache.pth",
    }

actual_diff_step = T // diffusion_args["skip_step"] + 1

if model_name == "TW_MultiSeq_Add":
    from Models import TW_MultiSeq_Add as TrajWeaver
    from Models import TW_MultiSeq_Add_Linkage as Linkage
    TW_args = {
        "in_c": 12,  # input trajectory encoding channels
        "out_c": 2,
        "diffusion_steps": T,  # maximum diffusion steps
        "c_list": [128, 128, 128, 256],  # channel schedule of stages, first element is stem output channels
        "blocks": ["RRRR", "RRRR", "RRRR"],  # number of resblocks in each stage
        "embed_c": 64,  # channels of mix embeddings
        "expend": 4,  # number of heads for attention
        "dropout": 0.0,  # dropout
    }

    link_args = {
        "max_t": T
    }
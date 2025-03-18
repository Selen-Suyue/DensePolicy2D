import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup
from termcolor import cprint

from policy import DSP
from dataset.realworld import RealWorldDataset, collate_fn
from utils.training import set_seed, plot_history, sync_loss

dataset = RealWorldDataset(
    path = "data/rise/pour_2",
    split = 'train',
    num_obs = 1,
    num_action =20,
    voxel_size = 0.005,
    aug = False,
    aug_jitter = False, 
    with_cloud = False,
    with_obj_action = True,
    no_project = True,
)

from tqdm import tqdm

batch_list = []

for i in tqdm(range(len(dataset))):
    dataitem = dataset[i]
    # let torch report 4 digits after dot
    torch.set_printoptions(precision=4, sci_mode=False)

    # print(dataitem['action'].shape)
    # print(torch.mean(dataitem['action'][:, :3], axis=0))
    # print(torch.mean(dataitem['action_normalized'][:, :3], axis=0))
    # print(torch.mean(dataitem['action_obj_normalized'][:, :3], axis=0))

    # input()
    batch_list.append(dataitem['action_normalized'][:, :3]) # [20, 3]

# stack at dim 0
batch_list = torch.stack(batch_list, dim=0) # [N, 20, 3]

# compute mean and std along N
mean = torch.mean(batch_list, axis=0)
std = torch.std(batch_list, axis=0)

print(mean)
print(std)

import copy
import glob
import math
import os
import time
from collections import deque
from distutils.dir_util import copy_tree

import numpy as np
import torch

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from envs.envs import make_vec_envs, get_vec_normalize
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot
from envs.pipelines import pipelines
from main import train_with_metric, execute_curriculum

args = get_args()

if __name__ == "__main__":
    pipeline_base = args.pipeline
    results = []
    tags = []
    for i in range(5):
        args.save_as = f"{2**i}cm"
        curr_pipeline = f"{args.pipeline}_{2**i}"
        if pipelines[curr_pipeline] is not None:
            tags += args.save_as
            results += train_with_metric(pipelines[curr_pipeline], execute_curriculum)
    averages = [np.mean(totals) for totals, _ in results]
    to_save = zip(tags, averages)
    for tag, average in to_save:
        print(f"{tag} - Average total training time {average}")
    torch.save(to_save, f"step_size_{args.pipeline}_final_results.pt")

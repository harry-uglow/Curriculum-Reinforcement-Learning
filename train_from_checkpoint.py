import os

import numpy as np
from torch import save

from envs.pipelines import pipelines
from main import train_with_metric, execute_curriculum, args

if __name__ == "__main__":
    pipeline_base = args.pipeline
    results = []
    tags = []
    save_base = args.save_as
    target = args.trg_succ_rate
    for i in range(5):
        length = f"{2**i}cm"
        curr_pipeline = f"{args.pipeline}_{2**i}"
        args.trg_succ_rate = target
        for seed in [0, 16, 32]:
            print(f"Training with {length} curriculum (if available)...")
            try:
                os.system(f"python main.py --save-as {save_base}_{length} --scene-name dish_rack "
                          f"--num-steps 256 --num-processes 16 --no-cuda --eval-interval 4 "
                          f"--initial-policy {save_base}_{length}_{seed}_dish_rack_11 "
                          f"--reuse-residual --trg-succ-rate {args.trg_succ_rate} "
                          f"--pipeline rack_res")
            except KeyError:
                continue


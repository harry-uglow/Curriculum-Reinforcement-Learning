import numpy as np
from torch import save

from a2c_ppo_acktr.arguments import get_args
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
        print(f"Training with {args.save_as} curriculum (if available)...")
        try:
            results += [train_with_metric(pipelines[curr_pipeline], execute_curriculum)]
            tags += [args.save_as]
        except KeyError:
            continue
    averages = [np.mean(totals) for totals, _ in results]
    to_save = list(zip(tags, averages))
    for tag, average in to_save:
        print(f"{tag} - Average total training time {average}")
    save(to_save, f"step_size_{args.pipeline}_final_results.pt")

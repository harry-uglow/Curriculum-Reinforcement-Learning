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
    curr_pipeline = f"{args.pipeline}_vis"
    args.trg_succ_rate = target
    print(f"Training with vis curriculum (if available)...")
    try:
        results += [train_with_metric(pipelines[curr_pipeline], execute_curriculum,
                                      f"{save_base}_vis")]
        tags += ['vis']
    except KeyError:
        pass
    averages = [np.mean(totals) for totals, _ in results]
    to_save = list(zip(tags, averages))
    for tag, average in to_save:
        print(f"{tag} - Average total training time {average}")
    save(to_save, f"{save_base}_step_size_{args.pipeline}_final_results.pt")

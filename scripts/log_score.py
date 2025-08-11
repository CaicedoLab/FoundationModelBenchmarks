import wandb
import os
import json
import subprocess
import polars as pl
import argparse
wandb.Settings(quiet=True)

# score_directory = "/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores/"
score_directory = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores'

def main(run_id, iteration, checkpoint_score_path):                
    score_csv = pl.read_csv(checkpoint_score_path) 
    allen_score = score_csv.filter(pl.col('dataset')=="Allen", pl.col("task")=="Task_two")['f1_score_macro'].sum()/3 # .item is eqv here
    hpa_score = score_csv.filter(pl.col('dataset')=="HPA", pl.col("task").is_in(['Task_two', "Task_three"]))['f1_score_macro'].sum()/6
    cp_score = score_csv.filter(pl.col('dataset')=="CP", pl.col("task").is_in(['Task_two', "Task_three", "Task_four"]))['f1_score_macro'].sum()/9
    
    chammi_score = allen_score + hpa_score + cp_score
    
    try:
        wandb.init(
            project="dinov2_chammi",
            id=run_id,
            resume="must",
            reinit='default'
        )
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")    
        wandb.log(
            {
                "eval/CHAMMI Score": chammi_score, 
                "eval/step": iteration
            }
        )

        wandb.finish()
    except:
        print("Score was not logged to wandb.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--run-id", type=str, required=True, help="ID of the run")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number to evaluate at")
    parser.add_argument("--score-path", type=str, required=True, help="Path to the CHAMMI score file")

    args = parser.parse_args()
    
    main(args.run_id, args.iteration, args.score_path)
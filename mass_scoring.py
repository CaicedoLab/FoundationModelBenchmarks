import os
import subprocess
import sys
from multiprocessing import Queue, Pool, Manager
from functools import partial
import argparse

# SCORES_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'
# ROOT_DIR = '/scr/data/CHAMMI/dataset'

SCORES_ROOT = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores'
FEATURES_ROOT    = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'
ROOT_DIR = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_metadata'

def main(dry_run: bool, log_off: bool):
    scored_models = os.listdir(SCORES_ROOT)
    evaled_models = os.listdir(FEATURES_ROOT)
    
    checkpoints_to_eval = []
    for model in evaled_models:            
        checkpoints:list[str] = os.listdir(os.path.join(FEATURES_ROOT, model))
        checkpoints = [tuple(checkpoint.rsplit('_', 1)) for checkpoint in checkpoints]
        checkpoints.sort(key=lambda x: int(x[-1]))
        for name, check in checkpoints:
            possible_score_path = os.path.join(SCORES_ROOT, model, f'{name}_{check}')
            if os.path.exists(possible_score_path) and f'{name}_{check}' in os.listdir(os.path.join(SCORES_ROOT, "logged", model)):
                continue
            dest_dir = os.path.join(SCORES_ROOT, model, f'{name}_{check}')
            feat_dir = os.path.join(FEATURES_ROOT, model, f'{name}_{check}')
            
            if dry_run:
                print(model, f'{name}_{check}')
            else:
                script = f"python morphem/benchmark.py --root-dir {ROOT_DIR} --dest-dir {dest_dir} --feature-dir {feat_dir} --feature-file pretrained_dinov2_vit_features.npy"                
                result = subprocess.run(
                    script,
                    shell=True,
                    capture_output=False,
                    check=True,
                    text=True
                )
            
                if not log_off:
                    score_path = os.path.join(dest_dir, 'knn_l2_full_results.csv')
                    log_script = f"python ./log_score.py --run-id {model} --iteration {check} --score-path {score_path}"                
                    result = subprocess.run(
                        log_script,
                        shell=True,
                        capture_output=False,
                        check=True,
                        text=True
                    )

    print(checkpoints_to_eval)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My program")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Perform a dry run without making actual changes")
    parser.add_argument('-l', "--log-off", action="store_true", help="Disable logging to wandb")
    args = parser.parse_args()
    
    main(args.dry_run, args.log_off)
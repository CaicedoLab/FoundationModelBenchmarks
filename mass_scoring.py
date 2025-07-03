import os
import subprocess
import sys
from multiprocessing import Queue, Pool, Manager
from functools import partial

SCORES_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores'
FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

def main():
    scored_models = os.listdir(SCORES_ROOT)
    evaled_models = os.listdir(FEATURES_ROOT)
    
    checkpoints_to_eval = []
    for model in evaled_models:            
        checkpoints:list[str] = os.listdir(os.path.join(FEATURES_ROOT, model))
        checkpoints = [tuple(checkpoint.rsplit('_', 1)) for checkpoint in checkpoints]
        checkpoints.sort(key=lambda x: int(x[-1]))
        for name, check in checkpoints:
            possible_score_path = os.path.join(SCORES_ROOT, model, f'{name}_{check}')
            if os.path.exists(possible_score_path) and f'{name}_{check}' in os.listdir(os.path.join(SCORES_ROOT, model)):
                continue
            
            script = f"python morphem/benchmark.py --root-dir /scr/data/CHAMMI/dataset --dest-dir /mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores/{model}/{name}_{check} --feature-dir /mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features/{model}/{name}_{check} --feature-file pretrained_dinov2_vit_features.npy"                
            result = subprocess.run(
                script,
                shell=True,
                capture_output=False,
                check=True,
                text=True
            )
            
            log_script = f"python ./log_score.py --run-id {model} --iteration {check} --score-path /mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores/{model}/{name}_{check}/knn_l2_full_results.csv"                
            result = subprocess.run(
                log_script,
                shell=True,
                capture_output=False,
                check=True,
                text=True
            )

    print(checkpoints_to_eval)
    
if __name__ == "__main__":
    main()
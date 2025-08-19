import os
import subprocess
import sys
from multiprocessing import Queue, Pool, Manager
from functools import partial
import argparse
from collections import defaultdict

SCORES_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores'
FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'
ROOT_DIR = '/scr/data/CHAMMI/dataset'

# CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/models'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/features'

# SCORES_ROOT = '/mnt/cephfs/mir/jcaicedo/projects//channel_vit_dinov1/scores'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/features'
# ROOT_DIR = '/scr/data/CHAMMI/dataset'

# SCORES_ROOT = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores'
# FEATURES_ROOT    = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'
# ROOT_DIR = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_metadata'

def get_dinov2_checkpoints(evaled_models: list):
    checkpoints_to_eval = []
    to_eval_score = defaultdict(list)
    
    for model in evaled_models:      
        if model == "logged":
            continue
        
        checkpoints:list[str] = os.listdir(os.path.join(FEATURES_ROOT, model))
        checkpoints = [tuple(checkpoint.rsplit('_', 1)) for checkpoint in checkpoints]

        checkpoints.sort(key=lambda x: int(x[-1]))
        for name, check in checkpoints:
            possible_score_path = os.path.join(SCORES_ROOT, model, f'{name}_{check}')
            if os.path.exists(possible_score_path) and f'{name}_{check}' in os.listdir(os.path.join(SCORES_ROOT, model)):
                continue
            dest_dir = os.path.join(SCORES_ROOT, model, f'{name}_{check}')
            feat_dir = os.path.join(FEATURES_ROOT, model, f'{name}_{check}')
            to_eval_score[model].append({
                'dest': dest_dir,
                'feat': feat_dir,
                'model': model,
                'check': check
            })
            
    return to_eval_score

def get_channelvit_checkpoints(evaled_models: list):
    scored_models = os.listdir(SCORES_ROOT)
    yet_to_score = filter(lambda x: x not in scored_models, evaled_models)
    
    to_eval_score = defaultdict(list)
    for model in yet_to_score:
        dest_dir = os.path.join(SCORES_ROOT, model)
        feat_dir = os.path.join(FEATURES_ROOT, model)
        to_eval_score[model].append({
                'dest': dest_dir,
                'feat': feat_dir,
                'model': model,
                'check': None
            })
    return to_eval_score

def main(dry_run: bool, log_off: bool, model_type: str):
    evaled_models = os.listdir(FEATURES_ROOT)
    
    if model_type == 'dinov2':
        to_eval_score = get_dinov2_checkpoints(evaled_models)
        feature_file = 'pretrained_dinov2_vit_features.npy'
    elif model_type == 'dinov1':
        to_eval_score = get_channelvit_checkpoints(evaled_models)
        feature_file = 'pretrained_vit_features.npy'
        
    for model in to_eval_score:
        checkpoints = to_eval_score[model]
        if model_type == 'dinov2':
            has_final_checkpoint = False
            for check in checkpoints:
                if 'final_model' in check['dest']:
                    has_final_checkpoint = True
        else:
            has_final_checkpoint = True
        
        if has_final_checkpoint:
            if dry_run:
                    print(model)
            else:
                if model_type == 'dinov2':
                    checkpoints = sorted(checkpoints, key=lambda x: int(x['check']))
                
                for check in checkpoints:
                    script = f"score --root-dir {ROOT_DIR} --dest-dir {check['dest']} --feature-dir {check['feat']} --feature-file {feature_file}"                
                    result = subprocess.run(
                        script,
                        shell=True,
                        capture_output=False,
                        check=True,
                        text=True
                    )
                
                    if not log_off and model_type == "dinov2":
                        score_path = os.path.join(check['dest'], 'knn_l2_full_results.csv')
                        log_script = f"python ./log_score.py --run-id {model} --iteration {check['check']} --score-path {score_path}"                
                        result = subprocess.run(
                            log_script,
                            shell=True,
                            capture_output=False,
                            check=True,
                            text=True
                        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk CHAMMI scoring")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Perform a dry run without making actual changes")
    parser.add_argument('-l', "--log-off", action="store_true", help="Disable logging to wandb, only used with dinov2 models. Ignored otherwise.")
    parser.add_argument('-m', '--model-type', default='dinov2', help='The type of model that can be evaluated. For channelvit use dionv1', choices=['dionv2', 'dinov1'])
    args = parser.parse_args()
    
    main(args.dry_run, args.log_off, args.model_type)

import os
import subprocess
import sys
from multiprocessing import Queue, Pool, Manager
from functools import partial
import argparse

# CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/baseline'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/testing_features'

CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/dino_artifacts'
FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

# CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/models'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/features'

def eval_checkpoint(q: Queue, model_check, model):
    model, checkpoint = model_check
    gpu = q.get()
    
    print(f"Evaluating {model} with GPU: {gpu}")
    
    if model == 'dinov2':
        result = subprocess.run(
                    f"extract --root-dir /scr/data/CHAMMI/dataset/ --feat-dir {os.path.join(FEATURES_ROOT, model)} --model dinov2 --model-size small --model-path {os.path.join(CHECKPOINTS_ROOT, model)} --gpu 0,1,2 --batch-size 32 --checkpoint {checkpoint} ",
                    shell=True,
                    # capture_output=True,
                    check=True,
                    text=True
                )
    
    else:
        result = subprocess.run(
                    f"extract --root-dir /scr/data/CHAMMI/dataset/ --feat-dir {os.path.join(FEATURES_ROOT, model)} --model channelvit --model-size small --model-path {checkpoint} --gpu 0,1,2 --batch-size 32 --checkpoint {checkpoint} ",
                    shell=True,
                    # capture_output=True,
                    check=True,
                    text=True
                )
    
    q.put(gpu)

def get_dinov2_checkpoints(models: list, evaled_models: list):
    checkpoints_to_eval = []
    for model in models:
        if 'eval' not in os.listdir(os.path.join(CHECKPOINTS_ROOT, model)):
            continue
        
        if 'original' not in model and 'noise' not in model:
            continue
        
        model_dir = os.path.join(CHECKPOINTS_ROOT, model) 
        eval_dir = os.path.join(model_dir, 'eval')
        checkpoints = os.listdir(eval_dir)
        
        if model in evaled_models:
            evaled_checkpoints = os.listdir(os.path.join(FEATURES_ROOT, model))
            non_eval_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint not in evaled_checkpoints]
            checkpoints_to_eval.extend([(model, checkpoint) for checkpoint in non_eval_checkpoints])
        else:
            checkpoints_to_eval.extend([(model, checkpoint) for checkpoint in checkpoints])
    return checkpoints_to_eval

def get_channelvit_checkpoints(models: list, evaled_models: list):
    checkpoints_to_eval = []
    for model in models:        
        if model in evaled_models:
            continue
        
        checkpoint = os.path.join(CHECKPOINTS_ROOT, model, 'checkpoint.pth') 
        checkpoints_to_eval.append((model, checkpoint))
    return checkpoints_to_eval

def main(dry_run: bool, model: str):
    models = os.listdir(CHECKPOINTS_ROOT)
    evaled_models = os.listdir(FEATURES_ROOT)

    if model == 'dinov2':
        checkpoints_to_eval = get_dinov2_checkpoints(models, evaled_models)
    else:
        checkpoints_to_eval = get_channelvit_checkpoints(models, evaled_models)

    if dry_run:
        list(map(print, checkpoints_to_eval))
        return
    
    num_gpu = 7        
    with Manager() as manager:
        q = manager.Queue()
        
        for gpu_id in range(num_gpu):
            q.put(gpu_id)
        
        with Pool(processes=num_gpu) as p:
            for model_check in checkpoints_to_eval:
                p.apply_async(eval_checkpoint, args=(q, model_check, model))
                
            p.close()
            p.join()
        print("Jobs finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Print what work will be done without executing.")
    parser.add_argument("-m", "--model", default="dinov2", choices=['dinov2', 'dinov1'],
                        help="What model to run with for feature extraction. use dionv1 with channelvit")
    
    args = parser.parse_args()
    
    main(args.dry_run, args.model)
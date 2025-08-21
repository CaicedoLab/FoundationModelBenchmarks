import os
import subprocess
import sys
from multiprocessing import Queue, Pool, Manager
from functools import partial

# CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/kept_checkpoints'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/models'
FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/features'

def eval_checkpoint(q: Queue, model_check):
    model, checkpoint = model_check
    gpu = q.get()
    
    print(f"Evaluating {model} with GPU: {gpu}")
    
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

def main():
    models = os.listdir(CHECKPOINTS_ROOT)
    evaled_models = os.listdir(FEATURES_ROOT)

    # checkpoints_to_eval = get_dinov2_checkpoints(models, evaled_models)
    checkpoints_to_eval = get_channelvit_checkpoints(models, evaled_models)
    
    num_gpu = 7        
    with Manager() as manager:
        q = manager.Queue()
        
        for gpu_id in range(num_gpu):
            q.put(gpu_id)
        
        with Pool(processes=num_gpu) as p:
            for model_check in checkpoints_to_eval:
                p.apply_async(eval_checkpoint, args=(q, model_check))
                
            p.close()
            p.join()
        print("Jobs finished.")

if __name__ == "__main__":
    main()
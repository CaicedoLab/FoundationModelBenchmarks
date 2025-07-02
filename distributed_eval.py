import os
import subprocess
import sys
from multiprocessing import Queue, Pool, Manager
from functools import partial

CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/kept_checkpoints'
FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

def eval_checkpoint(q: Queue, model_check):
    model, checkpoint = model_check
    gpu = q.get()
    print(f"Evaluating {model} at checkpoint {checkpoint} with GPU: {gpu}")
    #condor_submit wandb_key={os.environ.get('WANDB_API_KEY')} model={os.path.join(CHECKPOINTS_ROOT, model)} checkpoint={checkpoint} output={os.path.join(FEATURES_ROOT, model, checkpoint)} condor_eval.sh
    result = subprocess.run(
                f"python morphem/feature_extraction.py --root-dir /scr/data/CHAMMI/dataset/ --feat-dir {os.path.join(FEATURES_ROOT, model, checkpoint)} --model dinov2 --model-size small --model-path {os.path.join(CHECKPOINTS_ROOT, model)} --gpu {gpu} --batch-size 128 --checkpoint {checkpoint} ",
                shell=True,
                capture_output=True,
                check=True,
                text=True
            )
    
    q.put(gpu)

def main():
    models = os.listdir(CHECKPOINTS_ROOT)
    evaled_models = os.listdir(FEATURES_ROOT)
    
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
            
    with Manager() as manager:
        q = manager.Queue()
        
        for gpu_id in range(7):
            q.put(gpu_id)
        
        print(q, "GPUs available")
        with Pool(processes=7) as p:
            for model_check in checkpoints_to_eval:
                p.apply_async(eval_checkpoint, args=(q, model_check))
                
            p.close()
            p.join()
        print("Jobs finished.")

if __name__ == "__main__":
    main()
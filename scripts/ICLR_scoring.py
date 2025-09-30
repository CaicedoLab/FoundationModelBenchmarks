import os
import subprocess
import argparse
from tqdm import tqdm
from multiprocessing import Queue, Pool, Manager
import subprocess

FEATURES_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/benchmarking/features'
SCORES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/benchmarkinf/scores'

def get_checkpoints():
    checkpoints = []
    for dirpath, dirnames, filenames in os.walk(FEATURES_ROOT):
        if len(filenames) == 0 and len(dirnames) == 3:
            checkpoints.append((dirpath, dirpath.replace('/features/', '/scores/')))
    return checkpoints

def score(q: Queue, feat_path, out_path):
    gpu = q.get()

    command = f"CUDA_VISIBLE_DEVICES={str(gpu)} score --root-dir /scr/data/CHAMMI/dataset --dest-dir {out_path} --feature-dir {feat_path} --feature-file pretrained_vit_features.npy --use-gpu"
    
    try:
        subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        check=True,
                        text=True
                    )
    except:
        print(f"MODEL {feat_path} FAILED!")
    q.put(gpu)

def main(dry_run: bool):
    checkpoints_to_eval = get_checkpoints()
    gpu = list(range(0,8))
    gpu = [*list(range(0,8)), *list(range(0,8)), *list(range(0,8))]
    with Manager() as manager:
        q = manager.Queue()
        
        for gpu_id in gpu:
            q.put(int(gpu_id))
        
        with Pool(processes=len(gpu)) as p:
            results = []
            for feat_dir, out_dir in checkpoints_to_eval:
                res = p.apply_async(score, args=(q, feat_dir, out_dir))
                results.append(res)
                
            for res in tqdm(results, desc="Extracting"):
                res.get() 
            
            p.close()
            p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true", help="Perform a dry run without making actual changes")
    args = parser.parse_args()
    
    main(args.dry_run)              

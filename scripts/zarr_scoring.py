import os
import subprocess
import argparse
from tqdm import tqdm
from multiprocessing import Queue, Pool, Manager
import subprocess

def get_checkpoints():
    basepaths = ['/mnt/cephfs/mir/jcaicedo/projects/parcha/27c8bf6_parcha_ngram']#, '/mnt/cephfs/mir/jcaicedo/projects/parcha/27c8bf6_parcha_boc']
    check_paths = []
    for path in basepaths:
        out_path = os.path.join(path, 'scores')
        path = os.path.join(path, 'features')
        checkpoints = os.listdir(path)
        for checkpoint in checkpoints:
            check_paths.append((os.path.join(path, checkpoint), os.path.join(out_path, checkpoint)))
    return check_paths

def score(q: Queue, feat_path, out_path):
    gpu = q.get()

    command = f"CUDA_VISIBLE_DEVICES={str(gpu)} score --root-dir /scr/data/CHAMMI/dataset --dest-dir {out_path} --feature-dir {feat_path} --feature-file pretrained_vit_features.npy --use-gpu"
    
    subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    check=True,
                    text=True
                )

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

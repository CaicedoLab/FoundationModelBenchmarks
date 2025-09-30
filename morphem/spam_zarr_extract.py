from tqdm import tqdm
import os
from multiprocessing import Queue, Pool, Manager
import subprocess

def extract_features(q: Queue, checkpoint):
    gpu = q.get()
    
    check, out, model = checkpoint
    
    command = f"CUDA_VISIBLE_DEVICES={str(gpu)} python temp_merge_extract.py --root-dir /scr/data/CHAMMI/dataset/ --out-dir {out} --model {model} --model-path {check} --batch-size 32 "
    
    subprocess.run(
                        command,
                        shell=True,
                        # capture_output=True,
                        check=True,
                        text=True
                    )

    q.put(gpu)

def main():
    basepaths = ['/mnt/cephfs/mir/jcaicedo/projects/parcha/27c8bf6_parcha_ngram']#, '/mnt/cephfs/mir/jcaicedo/projects/parcha/27c8bf6_parcha_boc']
    check_paths = []
    for path in basepaths:
        model = 'dinov2'
        if path.endswith('ngram'):
            model = 'ngram'
        
        out_path = os.path.join(path, 'features')
        path = os.path.join(path, 'checkpoints')
        checkpoints = os.listdir(path)
        for checkpoint in checkpoints:
            check_paths.append((os.path.join(path, checkpoint), os.path.join(out_path, checkpoint.split('.')[0]), model))

    gpu = list(range(0,8))
    with Manager() as manager:
        q = manager.Queue()
        
        for gpu_id in gpu:
            q.put(int(gpu_id))
        
        with Pool(processes=len(gpu)) as p:
            results = []
            for checkpoint in check_paths:
                res = p.apply_async(extract_features, args=(q, checkpoint))
                results.append(res)
                
            for res in tqdm(results, desc="Extracting"):
                res.get() 
            
            p.close()
            p.join()
            


def path_expansion(path: str):
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))

if __name__ == "__main__":
    main()
    

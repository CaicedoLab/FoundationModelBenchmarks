import os
import subprocess
import sys
import argparse

CHECKPOINTS_ROOT = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/dino_artifacts'
FEATURES_ROOT    = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

# CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/kept_checkpoints'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

def main(dry_run: bool):
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

    for (model, checkpoint) in checkpoints_to_eval:   
        wandb_key =  os.environ.get('WANDB_API_KEY')
        
        output_dir = os.path.join(FEATURES_ROOT, model, checkpoint)
        if dry_run:
            print(model, checkpoint)
        else:
            submit_cmd = f"condor_submit wandb_key={wandb_key} model_path={os.path.join(CHECKPOINTS_ROOT, model)} checkpoint={checkpoint} feature_out={output_dir} condor_eval.sh"
            
            result = subprocess.run(
                        submit_cmd,
                        shell=True,
                        capture_output=True,
                        check=True,
                        text=True
                    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My program")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Perform a dry run without making actual changes")
    args = parser.parse_args()
    
    main(args.dry_run)

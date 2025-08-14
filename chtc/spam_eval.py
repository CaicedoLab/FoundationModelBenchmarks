import os
import subprocess
import sys
import argparse

# CHECKPOINTS_ROOT = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/dino_artifacts'
# FEATURES_ROOT    = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

# CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/kept_checkpoints'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

# CHECKPOINTS_ROOT = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/models'
# FEATURES_ROOT    = '/mnt/cephfs/mir/jcaicedo/projects/channel_vit_dinov1/features'

CHECKPOINTS_ROOT = '/hdd/jcaicedo/projects/channel_vit_dinov1/models'
FEATURES_ROOT    = '/hdd/jcaicedo/projects//channel_vit_dinov1/features'


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

def main(dry_run: bool, model_type: str, model_size: str, gpus: str, features: str, checkpoints: str):   
    models = os.listdir(checkpoints)
    evaled_models = os.listdir(features)
    
    if model_type == 'dinov2':
        checkpoints_to_eval = get_dinov2_checkpoints(models, evaled_models)
    elif model_type == 'channelvit':
        checkpoints_to_eval = get_channelvit_checkpoints(models, evaled_models)
    else:
        raise NotImplementedError("Model type supplied is not one of the available options")

    for (model, checkpoint) in checkpoints_to_eval:   
        wandb_key =  os.environ.get('WANDB_API_KEY')
        
        if 'ngram' in model:
            continue
        
        output_dir = os.path.join(features, model, checkpoint)
        if dry_run:
            print(model, checkpoint)
        else:
            submit_cmd = f"condor_submit wandb_key={wandb_key} model_type={model_type} model_size={model_size} gpus={gpus} model_path={os.path.join(checkpoints, model)} checkpoint={checkpoint} feature_out={output_dir} req_gpu={len(gpus.split(','))} condor_eval.sh"
            
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
    parser.add_argument("-m", "--model-type", default='dinov2', help="Denote what model is to extract features with")
    parser.add_argument("-s", "--model-size", default='small', help="Denote what model size is to extract features with. Not all models need this")
    parser.add_argument("-g", "--gpus", default='0', help="Comme separated gpu numbers. Examples are 0,1,2 or 0, or 0,3 etc")
    parser.add_argument("-f", "--features", default=FEATURES_ROOT, help="Path to where the output features should be stored")
    parser.add_argument("-c", "--checkpoints", default=CHECKPOINTS_ROOT, help="Path to where the checkpoints to be potentially evaluated are")
    args = parser.parse_args()
    
    main(args.dry_run, args.model_type, args.model_size, args.gpus, args.features, args.checkpoints)

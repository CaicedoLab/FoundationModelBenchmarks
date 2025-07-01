import os
import subprocess

CHECKPOINTS_ROOT = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/dino_artifacts'
FEATURES_ROOT    = '/hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_features'

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
            
    for (model, checkpoint) in checkpoints_to_eval:    
        result = subprocess.run(
                    f"condor_submit wandb_key={os.environ.get('WANDB_API_KEY')} model={os.path.join(CHECKPOINTS_ROOT, model)} checkpoint={checkpoint} output={os.path.join(FEATURES_ROOT, model, checkpoint)} condor_eval.sh",
                    shell=True,
                    capture_output=True,
                    check=True,
                    text=True
                )

if __name__ == "__main__":
    main()
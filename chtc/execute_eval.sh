mkdir FoundationModels/FoundationModels
mv FoundationModels/* FoundationModels/FoundationModels
mv dinov2 FoundationModels/
cd FoundationModels
PYTHONPATH="./dinov2/:../FoundationModels" torchrun --standalone --nnodes=1 --nproc-per-node=8 ./dinov2/dinov2/train/train.py --config-file ./dinov2/dinov2/configs/train/vit_single_channel.yaml --output-dir /hdd/jcaicedo/projects/foundation_models_and_benchmarking/dino_artifacts  train.name="$RUN_NAME" optim.adamw_beta2=$BETA_2

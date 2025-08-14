. /etc/profile.d/pixi.sh 
unzip ./chammi_dataset.zip
mkdir ./temp_feats
python morphem/feature_extraction.py --root-dir ./dataset/ --feat-dir ./temp_feats --model $MODEL_TYPE --model-size $MODEL_SIZE --model-path $MODEL_PATH --gpu $GPUS --batch-size 32 --checkpoint $CHECKPOINT  
mkdir -p $FEATURE_DIR
mv ./temp_feats/* $FEATURE_DIR
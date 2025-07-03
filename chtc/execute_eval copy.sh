. /etc/profile.d/pixi.sh 
unzip ./chammi_dataset.zip
mkdir ./temp_feats
python morphem/feature_extraction.py --root-dir ./dataset/ --feat-dir ./temp_feats --model dinov2 --model-size small --model-path $MODEL_PATH --gpu 0 --batch-size 128 --checkpoint $CHECKPOINT  
mkdir -p $FEATURE_DIR
mv ./temp_feats/* $FEATURE_DIR
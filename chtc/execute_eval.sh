. /etc/profile.d/pixi.sh 
unzip ./chammi_dataset.zip
python morphem/feature_extraction.py --root-dir ./dataset/ --feat-dir $OUTPUT --model dinov2 --model-size small --model-path $MODEL --gpu 0 --batch-size 128 --checkpoint $CHECKPOINT 
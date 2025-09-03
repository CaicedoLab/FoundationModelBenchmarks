rm -r /mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/testing_scores/*
rm -r /mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/testing_features/*

pixi run extract
pixi run score
pixi run print
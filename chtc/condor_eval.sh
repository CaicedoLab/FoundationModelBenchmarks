container_image = file:///staging/groups/caicedo_group/images/feat_extract.sif
log = train$(Cluster).log
universe = container
executable = execute_eval.sh
arguments = $(Process)
output = train$(Cluster)_$(Process).out
error = train$(Cluster)_$(Process).err
environment = "WANDB_API_KEY=$(wandb_key) MODEL=$(model) CHECKPOINT=$(checkpoint) OUTPUT=$(output)"

# Specify that HTCondor should transfer files to and from the
#  computer where each job runs. The last of these lines *would* be
#  used if there were any other files needed for the executable to use.
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_input_files = execute_eval.sh, /home/jgpeters3/FoundationModelBenchmarks/morphem, /hdd/jcaicedo/projects/foundation_models_and_benchmarking/chammi_dataset.zip
# Tell HTCondor what amount of compute resources 
#  each job will need on the computer where it runs.
# Machine == "jcaicedogpu0000.chtc.wisc.edu" || Machine == "jcaicedogpu0001.chtc.wisc.edu" || Machine == "jcaicedogpu0002.chtc.wisc.edu" ||
requirements = (Machine == "jcaicedogpu0001.chtc.wisc.edu" || Machine == "jcaicedogpu0002.chtc.wisc.edu" || Machine == "coba2000.chtc.wisc.edu")
request_cpus = 12
request_memory = 64GB
request_disk =  96GB
request_gpus = 1
queue 1

# run this file with condor_submit wandb_key=$WANDB_API_KEY batch=NUMBER lr=NUMBER name=NAME hyperparam_sweep.sh
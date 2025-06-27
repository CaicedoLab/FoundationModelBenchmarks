import time
import zmq
import wandb
import os
import json
import subprocess
import polars as pl

wandb.Settings(quiet=True)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

score_directory = "/mnt/cephfs/mir/jcaicedo/projects/foundation_models_and_benchmarking/chammi_scores/"

while True:
    #  Wait for next request from client
    message = socket.recv()
    message = json.loads(bytes.decode(message))
    print("Received request: %s" % str(message))
    socket.send(b"o7")
    
    # {message['run_id']}
    result = subprocess.run(
        f"pixi run score vit_s_146_batch",
        shell=True,
        capture_output=True,
        check=True,
        text=True
    )
    # message['run_id']
    score_path = os.path.join(score_directory, "vit_s_146_batch", "knn_l2_full_results.csv")
    
    score_csv = pl.read_csv(score_path) 
    allen_score = score_csv.filter(pl.col('dataset')=="Allen", pl.col("task")=="Task_two")['f1_score_macro'].sum()/3 # .item is eqv here
    hpa_score = score_csv.filter(pl.col('dataset')=="HPA", pl.col("task").is_in(['Task_two', "Task_three"]))['f1_score_macro'].sum()/6
    cp_score = score_csv.filter(pl.col('dataset')=="CP", pl.col("task").is_in(['Task_two', "Task_three", "Task_three"]))['f1_score_macro'].sum()/9
    
    chammi_score = allen_score + hpa_score + cp_score
    print(chammi_score)
    try:
        wandb.init(
            project="dinov2_chammi",
            id=message['run_id'],
            resume="must",
            reinit='default'
        )
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")    
        wandb.log(
            {
                "eval/CHAMMI Score": chammi_score, 
                "eval/step": message['iteration']
            }
        )

        wandb.finish()
    except:
        print("Score was not logged to wandb.")

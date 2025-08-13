
import pandas as pd
import os
import json
import argparse
from evaluation import evaluate, create_umap
import wandb

import warnings
warnings.filterwarnings("ignore")

def save_results(results, dest_dir, dataset, classifier, knn_metric):
    # Helper function
    # Save results for each dataset as a json dictionary at dest_dir
    full_reports_dict = {}
    full_reports_dict['target_encoding'] = results["encoded_target"]
    for task_ind, task in enumerate(results["tasks"]):
        full_reports_dict[task] = results["reports_dict"][task_ind]

    if not os.path.exists(dest_dir+ '/'):
        os.makedirs(dest_dir+ '/')
    
    if classifier == 'knn':
        dict_path = f'{dest_dir}/{dataset}_{classifier}_{knn_metric}_results.json'
    else:
        dict_path = f'{dest_dir}/{dataset}_{classifier}_results.json'
        
    with open(dict_path, 'w') as f:
        json.dump(full_reports_dict, f)

    return
            
def run_benchmark(root_dir, dest_dir, feature_dir, feature_file, classifier='knn', umap=False, use_gpu=True, knn_metric='l2'):

    # encode dataset, task, and classifier
    task_dict = pd.DataFrame({'dataset':['Allen', 'HPA', 'CP'], 
                              'classifier':[classifier for i in range(3)], \
                              'leave_out': [None, 'Task_three', 'Task_four'], \
                              'leaveout_label': [None, 'cell_type', 'Plate'], \
                              'umap_label': ['Structure', 'cell_type', 'source'] 
                             })
    
    full_result_df = pd.DataFrame(columns=['dataset', 'task', 'classifier', 'accuracy', 'f1_score_macro'])
    
    # Iterrate over each dataset
    for idx, row in task_dict.iterrows():
        dataset        = row.dataset
        classifier     = row.classifier
        leave_out      = row.leave_out
        leaveout_label = row.leaveout_label
        umap_label     = row.umap_label
        
        features_path  = f'{feature_dir}/{dataset}/{feature_file}'
        df_path        = f'{root_dir}/{dataset}/enriched_meta.csv'
        
        # Create umap and run classification
        if umap:
            create_umap(dataset, 
                        features_path, 
                        df_path, 
                        dest_dir, 
                        ['Label', umap_label])
            
        results = evaluate(features_path, 
                           df_path, 
                           leave_out, 
                           leaveout_label, 
                           classifier, 
                           use_gpu, 
                           knn_metric)

        # Print the full results
        print('Results:')
        for task_ind, task in enumerate(results["tasks"]):
            print(f'Results for {dataset} {task} with {classifier} :')
            print(results["reports_str"][task_ind])
        
        # Save results as dictionary
        save_results(results, dest_dir, dataset, classifier, knn_metric)
        
        # Save results as csv
        result_temp = pd.DataFrame({'dataset': [dataset for i in range(len(results["tasks"]))],\
                        'task': results["tasks"],'classifier': [classifier for i in range(len(results["tasks"]))],\
                        'accuracy': results["accuracies"],'f1_score_macro': results["f1scores_macro"]})
        full_result_df = pd.concat([full_result_df, result_temp]).reset_index(drop=True)
    
    save_path = ''
    if classifier == 'knn':   
        save_path = f'{dest_dir}/{classifier}_{knn_metric}_full_results.csv'     
        full_result_df.to_csv(save_path, index=False) 
    else:
        save_path = f'{dest_dir}/{classifier}_full_results.csv'
        full_result_df.to_csv(save_path, index=False)

        
    return full_result_df

def main():
    root_dir, dest_dir, feature_dir, feature_file, classifier, umap, use_gpu, knn_metric = get_args()
    run_benchmark(root_dir, dest_dir, feature_dir, feature_file, classifier, umap, use_gpu, knn_metric)
    
def get_args():
    parser = argparse.ArgumentParser(description="Benchmark Args")
    parser.add_argument('--root-dir', type=str,
                        help='The root directory for data processing.')
    parser.add_argument('--dest-dir', type=str,
                        help='The destination directory for output files.')
    parser.add_argument('--feature-dir', type=str,
                        help='The directory containing feature files.')
    parser.add_argument('--feature-file', type=str,
                        help='The name of the feature file.')
    parser.add_argument('--classifier', type=str, default='knn',
                        choices=['knn'],
                        help='The classification algorithm to use. Default: %(default)s')
    parser.add_argument('--umap', action='store_true',
                        help='Enable UMAP dimensionality reduction. Default: %(default)s')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Enable GPU usage for processing. Default: %(default)s')
    parser.add_argument('--knn-metric', type=str, default='l2',
                        help='The metric to use for KNN classification. Default: %(default)s')

    args = parser.parse_args()
    return args.root_dir, args.dest_dir, args.feature_dir, args.feature_file, args.classifier, args.umap, args.use_gpu, args.knn_metric


if __name__ == "__main__":
    main()
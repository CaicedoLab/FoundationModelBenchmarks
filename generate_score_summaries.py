import argparse
import os
import polars as pl
from datetime import datetime
import csv

def main(
        input_directory:str, 
        dataset:str, 
        training_style:str, 
        model:str, 
        pipeline:str, 
        is_student:bool, 
        dry_run:bool, 
        output:str
    ):
    
    if is_student:
        model_type = "teacher"
    else:
        model_type = "student"
    
    score_files = []
    for root, dirs, files in os.walk(input_directory):
        if 'final_model' in root and 'logged' not in root:
            score_files.append(os.path.join(root, 'knn_l2_full_results.csv'))
    
    date = datetime.now().strftime("%#m/%#d/%Y")
    
    output_tsv = []
    for score_file in score_files: 
        model_arch = score_file.split('/')[-3]
        if dry_run:
            print(model_arch)
            continue
        
        score_df = pl.read_csv(score_file)
        scores = score_df['f1_score_macro'].to_list()
        row_entry = [0, dataset, training_style, date, model_arch, model, 0, 0, 0, 0, 0, *scores, pipeline, 0, 0, model_type]
        output_tsv.append(row_entry)
        
    pl.from_records(output_tsv, orient="row").write_csv(file=output, include_header=False, separator='\t')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files with various options.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input directory to process.")
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Print what work will be done without executing.")
    parser.add_argument("-o", "--output", type=str,
                        help="File to output data to.", default='./scores.tsv')
    parser.add_argument("--dataset", default='CHAMMIv1', type=str, help="Specify the dataset to use.")
    parser.add_argument("--training-style", default='Self-Supervised', type=str, help="Specify the training style.")
    parser.add_argument("--model", default='DINOV2', type=str, help="Specify the model to use.")
    parser.add_argument("--pipeline", default='John', type=str, help="Specify the pipeline to use.")
    parser.add_argument("--is-student", action="store_true", help="Say it was a student eval.")
    
    args = parser.parse_args()
    
    input_directory = os.path.abspath(os.path.expanduser(args.input))
    main(input_directory, args.dataset, args.training_style, args.model, args.pipeline, args.is_student, args.dry_run, args.output)
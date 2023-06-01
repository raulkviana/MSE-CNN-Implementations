"""@package docstring 

@file eval_model_stg6_metrics.py 

@brief Evaluates stage 6 
 
@section libraries_eval_model_stg6_metrics Libraries 
- sklearn.metrics
- MSECNN
- torch.utils.data
- torch
- argparse
- torch.utils.tensorboard
- datetime
- train_model_utils
- utils
- constants
- CustomDataset
- sys
- time
- matplotlib.pyplot

@section classes_eval_model_stg6_metrics Classes 
- None

@section functions_eval_model_stg6_metrics Functions 
- test(dataloader, model, device, loss_name)
- val_setup(dataloader_val, model, device)
- main()
 
@section global_vars_eval_model_stg6_metrics Global Variables 
- parser 
- args 
- loss_threshold 
- batch_size 
- qp 
- device 
- n_mod 
- num_workers 
- rs 
- writer 
- l_path_val 
- files_mod_name_stats 

@section todo_eval_model_stg6_metrics TODO 
- None 

@section license License 
MIT License 
Copyright (c) 2022 Raul Kevin do Espirito Santo Viana
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@section author_eval_model_stg6_metrics Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:23:10.679038
"""


"""
    Evaluate model for a given dataset, returns a csv file with the results
"""

# Imports
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import time
import argparse
from torch.utils.tensorboard import SummaryWriter 
import datetime
import matplotlib.pyplot as plt
import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import utils
    import MSECNN
    import CustomDataset
    import train_model_utils
    import constants
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")


# Parse arguments
parser = argparse.ArgumentParser(description=constants.script_description)
parser.add_argument("--batch", type=int)
parser.add_argument("--dev", type=int)
parser.add_argument("--workers", type=int)
parser.add_argument("--nmod", type=str)
parser.add_argument("--labels", type=float)

# Get arguments
args = parser.parse_args()

# Parameters
loss_threshold = float("-inf")
qp = 32
batch_size = args.batch  # No paper 32
device = args.dev
num_workers = args.workers
n_mod = args.nmod
l_path_val = args.labels

# Multi-thresholding coefficients for medium mode
rs = [0.3, 0.5]

print("Using {} device".format(device))
# Tensorboard variable
writer = SummaryWriter("runs/MSECNN_Eval_"+n_mod)

# Build name modifier
files_mod_name_stats = "_multi_batch_val_batch_{batch}_QP_{QP}_{nmod}".format(batch=batch_size, QP=qp, nmod=n_mod)

def test(dataloader, model, device, loss_name):
    # Initialize variables
    size = len(dataloader.dataset)
    size = int(size/batch_size)
    # History variables
    predictions = []
    ground_truths = []
    pred_vector = []
    ground_truths_vector = []

    # Select model mode
    for m in model:
        m.eval()

    # With no gradient descent
    with torch.no_grad():

        for i_batch, sample_batch in enumerate(dataloader):
            # Obtain data for training
            CTU = torch.reshape(sample_batch[0], shape=(-1, 1, 128, 128))
            cu_pos_stg2 = torch.reshape(sample_batch[1], shape=(-1, 2))
            cu_pos_stg3 = torch.reshape(sample_batch[2], shape=(-1, 2))
            cu_pos_stg4 = torch.reshape(sample_batch[3], shape=(-1, 2))
            cu_pos_stg5 = torch.reshape(sample_batch[4], shape=(-1, 2))
            cu_pos = torch.reshape(sample_batch[5], shape=(-1, 2))
            cu_size_stg2 = torch.reshape(sample_batch[6], shape=(-1, 2))
            cu_size_stg3 = torch.reshape(sample_batch[7], shape=(-1, 2))
            cu_size_stg4 = torch.reshape(sample_batch[8], shape=(-1, 2))
            cu_size_stg5 = torch.reshape(sample_batch[9], shape=(-1, 2))
            cu_size = torch.reshape(sample_batch[10], shape=(-1, 2))
            split_stg2 = torch.reshape(sample_batch[11], (-1, 1))
            split_stg3 = torch.reshape(sample_batch[12], (-1, 1))
            split_stg4 = torch.reshape(sample_batch[13], (-1, 1))
            split_stg5 = torch.reshape(sample_batch[14], (-1, 1))
            split = torch.reshape(sample_batch[15], (-1, 1))
            RDs = torch.reshape(sample_batch[16], shape=(-1, 6)) 

            # Convert type and change device
            RDs = RDs.to(dtype=torch.float64).to(device)
            CUs = CTU.to(device)
            Y = split
            Y = train_model_utils.one_hot_enc(Y).to(device)

            # Compute prediction
            # Stage 1 and 2
            pred_stg2, CUs, ap = model[0](CUs, cu_size_stg2, cu_pos_stg2)  # Pass CU through network
            
            # Stage 3
            pred_stg3, CUs, ap = model[1](CUs, ap, split_stg2, cu_size_stg3, cu_pos_stg3)   # Pass CU through network

            # Stage 4
            pred_stg4, CUs, ap = model[2](CUs, ap, split_stg3, cu_size_stg4, cu_pos_stg4)   # Pass CU through network

            # Stage 5
            pred_stg5, CUs, ap = model[3](CUs, ap, split_stg4, cu_size_stg5, cu_pos_stg5)   # Pass CU through network

            # Stage 6
            pred, CUs, ap = model[4](CUs, ap, split_stg5, cu_size, cu_pos)   # Pass CU through network

            # Obtain results in different format
            pred_num = train_model_utils.obtain_mode(pred)
            Y_num = train_model_utils.obtain_mode(Y)

            # Update lists
            predictions.extend(pred_num.tolist())
            ground_truths.extend(Y_num.tolist())
            ground_truths_vector.extend(Y.tolist())
            pred_vector.extend(pred.tolist())

            # Print information about training
            if (i_batch+1) % 10 == 0:
                utils.echo("Complete:{percentage:.0%}".format(percentage=i_batch / size))

            # Delete later condition
            # if i_batch == 10:
            #    break

    return predictions, ground_truths, pred_vector, ground_truths_vector

def val_setup(dataloader_val, model, device):
    # Print current time
    train_model_utils.print_current_time()

    # Cycle start time
    start = time.time()

    # Validation
    print("VALIDATING")
    # Validate on validation data
    predictions, ground_truths, pred_vector, ground_truths_vector = test(dataloader_val, model, device, "val")

    # Print information about training
    print()
    print("Validation in validation data:")
    print(classification_report(ground_truths, predictions))
    print()

    # Compute metrics from validation
    f1_test, recall_test, precision_test, accuracy_test =\
            train_model_utils.model_simple_metrics(predictions, ground_truths)

    # Compute ROC curve
    ROC_test = train_model_utils.compute_ROC_curve(pred_vector, ground_truths_vector, predictions)
    # ROC curve: Training data
    writer.add_figure("rocCurve/eval", ROC_test, 0)
    plt.clf()
    # Compute confusion matrix
    conf_mat_test = train_model_utils.compute_conf_matrix(predictions, ground_truths)
    # Confusion matrix: Training data
    writer.add_figure("confMatrix/eval", conf_mat_test, 0)
    # Compute top k accuracy
    top_2_accuracy = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector), torch.tensor(ground_truths), 2)
    top_3_accuracy = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector), torch.tensor(ground_truths), 3)
    top_4_accuracy = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector), torch.tensor(ground_truths), 4)
    # Compute multi-thresholdin performance
    pred_mult_thres_0_3 = utils.multi_thresholding(rs[0], torch.tensor(pred_vector))
    mul_thre_perfo_0_3 = train_model_utils.compute_multi_thres_performance(pred_mult_thres_0_3, ground_truths)
    pred_mult_thres_0_5 = utils.multi_thresholding(rs[1], torch.tensor(pred_vector))
    mul_thre_perfo_0_5 = train_model_utils.compute_multi_thres_performance(pred_mult_thres_0_5, ground_truths)
    # Mean splits sent
    mean_splits_sent_0_3 = train_model_utils.compute_num_splits_sent(pred_mult_thres_0_3)
    mean_splits_sent_0_5 = train_model_utils.compute_num_splits_sent(pred_mult_thres_0_5)
    # Save metric from training
    writer.add_scalars("metrics/eval", {"f1-score": f1_test, "recall": recall_test, "precision": precision_test, "accuracy": accuracy_test, "top_3": top_3_accuracy, "top_2": top_2_accuracy, "top_4": top_4_accuracy, "multi-thres 0.3": mul_thre_perfo_0_3, "multi-thres 0.5": mul_thre_perfo_0_5, "mean splits sent 0.3": mean_splits_sent_0_3, "mean splits sent 0.5": mean_splits_sent_0_5}, 0)

    end = time.time()
    print("Time elapsed in this epoch:", str(datetime.timedelta(seconds=end-start)))

    # Print current time
    train_model_utils.print_current_time()
    print("Done!")

    # Make sure all events have been written to the disk
    writer.flush()
    writer.close() # Close session

    return model


def main():

    # Initialize Model
    stg1_2 = MSECNN.MseCnnStg_1(device=device, QP=qp).to(device)
    stg3 = MSECNN.MseCnnStg_x(device=device, QP=qp).to(device)
    stg4 = MSECNN.MseCnnStg_x(device=device, QP=qp).to(device)
    stg5 = MSECNN.MseCnnStg_x(device=device, QP=qp).to(device)
    stg6 = MSECNN.MseCnnStg_x(device=device, QP=qp).to(device)
    model = (stg1_2, stg3, stg4, stg5, stg6)

    ans = str(input('Do you want to load any existing model? Y/N \n'))
    if ans == 'Y' or ans == 'y':
       path = "best_stg6_multi_batch_iter_100_batch_32_QP_32_beta_0.0_lr_0.0006_stg6_no_beta"
       model = train_model_utils.load_model_parameters_eval(model, path, device)

    # Prepare testing data, Dataset and Dataloader
    val_data = CustomDataset.CUDatasetStg6Compl(files_path=l_path_val)
    batch_sampler_val = CustomDataset.SamplerStg6(val_data, batch_size)  # Batch Sampler
    dataloader_val = DataLoader(val_data, num_workers=num_workers, batch_sampler=batch_sampler_val)
    
    # Train
    print("Starting training...")
    val_setup(dataloader_val, model, device)

if __name__ == "__main__":
    main()

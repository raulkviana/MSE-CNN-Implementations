"""@package docstring 

@file eval_model_stg6.py 

@brief Evaluates stage 6, outputs predictions and corresponding ground-truth in a .csv file 
 
@section libraries_eval_model_stg6 Libraries 
- MSECNN
- torch.utils.data
- argparse
- datetime
- train_model_utils
- utils
- dataset_utils
- constants
- CustomDataset
- sys
- time

@section classes_eval_model_stg6 Classes 
- None 

@section functions_eval_model_stg6 Functions 
- test(dataloader, model, device, loss_name)
- val_setup(dataloader_val, model, device)
- main()
 
@section global_vars_eval_model_stg6 Global Variables 
- parser 
- args 
- loss_threshold 
- batch_size 
- qp 
- device 
- n_mod 
- num_workers 
- l_path_val 
- files_mod_name_stats 

@section todo_eval_model_stg6 TODO 
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

@section author_eval_model_stg6 Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:23:10.677042
"""

# Importsimport torch
from torch.utils.data import DataLoader
import argparse
import datetime
import time
import sys
import torch
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import utils
    import msecnn
    import custom_dataset
    import train_model_utils
    import constants
    import dataset_utils
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")

# Parse arguments
parser = argparse.ArgumentParser(description=constants.script_description)
parser.add_argument("--batch", type=int)
parser.add_argument("--dev", type=int)
parser.add_argument("--workers", type=int)
parser.add_argument("--nmod", type=str)
parser.add_argument("--thres", type=float)
parser.add_argument("--labels", type=float)

# Get arguments
args = parser.parse_args()

# Parameters
loss_threshold = float("-inf")
qp = 32
batch_size = args.batch
device = args.dev
num_workers = args.workers
rs = args.thres
n_mod = args.nmod
l_path_val = args.labels

# Get arguments
args = parser.parse_args()

print("Using {} device".format(device))

# Build name modifier
files_mod_name_stats = "_multi_batch_val_batch_{batch}_QP_{QP}".format(batch=batch_size, QP=qp)

def test(dataloader, model, device, loss_name):
    # Initialize variables
    size = len(dataloader.dataset)
    size = int(size/batch_size)
    # History variables
    predictions = []
    ground_truths = []
    pred_vector = []
    ground_truths_vector = []
    orig_pos_x_lst = []
    orig_pos_y_lst = []
    orig_size_h_lst = []
    orig_size_w_lst = []
    POC_lst = []
    pic_name_lst = []
    
    total_time = 0

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
            # Obtain other info
            orig_pos_x = torch.reshape(sample_batch[17], shape=(-1, 1))
            orig_pos_y = torch.reshape(sample_batch[18], shape=(-1, 1))
            orig_size_h = torch.reshape(sample_batch[19], shape=(-1, 1))
            orig_size_w = torch.reshape(sample_batch[20], shape=(-1, 1))
            POC = torch.reshape(sample_batch[21], shape=(-1, 1))
            pic_name = sample_batch[22] 

            # Convert type and change device
            RDs = RDs.to(dtype=torch.float64).to(device)
            CUs = CTU.to(device)
            Y = split
            Y = train_model_utils.one_hot_enc(Y).to(device)

            # Initial time
            t0 = time.time()

            # Compute prediction
            # Stage 1 and 2
            pred_stg2, CUs, ap = model[0](CUs, cu_size_stg2, cu_pos_stg2)  # Pass CU through network
            
            # Stage 3
            pred_stg3, CUs, ap = model[1](CUs, ap, split_stg2, cu_size_stg3, cu_pos_stg3)   # Pass CU through network --- You can also use the predicitons of the previous stage

            # Stage 4
            pred_stg4, CUs, ap = model[2](CUs, ap, split_stg3, cu_size_stg4, cu_pos_stg4)   # Pass CU through network --- You can also use the predicitons of the previous stage

            # Stage 5
            pred_stg5, CUs, ap = model[3](CUs, ap, split_stg4, cu_size_stg5, cu_pos_stg5)   # Pass CU through network --- You can also use the predicitons of the previous stage

            # Stage 6
            pred, CUs, ap = model[4](CUs, ap, split_stg5, cu_size, cu_pos)   # Pass CU through network --- You can also use the predicitons of the previous stage

            # Compute time to process CTU
            total_time += (time.time()-t0)

            # Obtain results in different format
            pred_num = train_model_utils.obtain_mode(pred)
            Y_num = train_model_utils.obtain_mode(Y)

            if len(pred_num) != 1:
                pred_num = pred_num.reshape(1, -1)

            # Update lists
            predictions.extend(pred_num.tolist())
            ground_truths.extend(Y_num.tolist())
            ground_truths_vector.extend(Y.tolist())
            pred_vector.extend(pred.tolist())
            orig_pos_x_lst.extend(orig_pos_x.tolist())
            orig_pos_y_lst.extend(orig_pos_y.tolist())
            orig_size_h_lst.extend(orig_size_h.tolist())
            orig_size_w_lst.extend(orig_size_w.tolist())
            POC_lst.extend(POC.tolist())
            pic_name_lst.extend(list(pic_name))

            # Print information about training
            if (i_batch+1) % 10 == 0:
                utils.echo("Complete: {percentage:.0%}".format(percentage=i_batch / size))
        
        print("Average time to process each CTU batch: ", total_time/(i_batch+1))
        print("Total time to process all CTUs: ", total_time)

    return predictions, ground_truths, pred_vector, ground_truths_vector, orig_pos_x_lst, \
           orig_pos_y_lst, orig_size_h_lst, orig_size_w_lst, POC_lst, pic_name_lst

def val_setup(dataloader_val, model, device):
    # Print current time
    train_model_utils.print_current_time()

    # Cycle start time
    start = time.time()

    # Validation
    print("VALIDATING")
    # Validate on validation data
    predictions, ground_truths, pred_vector, ground_truths_vector, orig_pos_x_lst, orig_pos_y_lst, \
    orig_size_h_lst, orig_size_w_lst, POC_lst, pic_name_lst = test(dataloader_val, model, device, "val")

    # Modify lists
    funct = lambda x: x[0]
    orig_pos_x_lst = list(map(funct, orig_pos_x_lst))
    orig_pos_y_lst = list(map(funct, orig_pos_y_lst))
    orig_size_h_lst = list(map(funct, orig_size_h_lst))
    orig_size_w_lst = list(map(funct, orig_size_w_lst))
    POC_lst = list(map(funct, POC_lst))

    # Compute multi-thresholdin 
    pred_mult_thres = utils.multi_thresholding(rs, torch.tensor(pred_vector))

    # Make csv file
    lst_lst = []
    lst_lst.append(ground_truths)
    lst_lst.append(pred_mult_thres)
    lst_lst.append(orig_pos_x_lst)
    lst_lst.append(orig_pos_y_lst)
    lst_lst.append(orig_size_h_lst)
    lst_lst.append(orig_size_w_lst)
    lst_lst.append(POC_lst)
    lst_lst.append(pic_name_lst)
    file_name = n_mod+"_results_stg_6_thres_"+str(rs)
    fields_names = ["split_gt","split_pred", "x", "y", "height", "width", "POC", "pic_name"]
    dataset_utils.lst2csv_v2(lst_lst, file_name, fields_names)

    end = time.time()
    print("Time elapsed in this epoch:", str(datetime.timedelta(seconds=end-start)))

    # Print current time
    train_model_utils.print_current_time()
    print("Done!")

    return model


def main():
    # Initialize Model
    stg1_2 = msecnn.MseCnnStg1(device=device, QP=qp).to(device)
    stg3 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
    stg4 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
    stg5 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
    stg6 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
    model = (stg1_2, stg3, stg4, stg5, stg6)

    ans = str(input('Do you want to load any existing model? Y/N \n'))
    if ans == 'Y' or ans == 'y':
       path = "coefficients/best_stg6_multi_batch_iter_100_batch_32_QP_32_beta_0.0_lr_0.0006_stg6_no_beta"
       model = train_model_utils.load_model_parameters_eval(model, path, device)

    # Prepare testing data, Dataset and Dataloader
    val_data = custom_dataset.CUDatasetStg6Compl(files_path=l_path_val)
    batch_sampler_val = custom_dataset.SamplerStg6(val_data, batch_size)  # Batch Sampler
    dataloader_val = DataLoader(val_data, num_workers=num_workers, batch_sampler=batch_sampler_val)

    # Train
    print("Starting training...")
    val_setup(dataloader_val, model, device)

if __name__ == "__main__":
    main()

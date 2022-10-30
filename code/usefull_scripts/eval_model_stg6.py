"""
    Evaluate model for a given dataset, returns a csv file with the results
"""

# Imports
from __future__ import print_function, division
from sqlalchemy import true
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from torch.utils.data import DataLoader
import MSECNN
import dataset_utils
import CustomDataset
import train_model_utils
from sklearn.metrics import classification_report
import constants
import utils
import time
import argparse
from torch.utils.tensorboard import SummaryWriter 
import datetime
import numpy as np
import pandas as pd
import random

# Parse arguments
parser = argparse.ArgumentParser(description=constants.script_description)
#parser.add_argument("--batch", type=int)
parser.add_argument("--dev", type=int)
parser.add_argument("--workers", type=int)
parser.add_argument("--nmod", type=str)
parser.add_argument("--thres", type=float)

# Get arguments
args = parser.parse_args()

# Parameters
loss_threshold = float("-inf")
qp = 32
batch_size = 1 # No paper 32
device = args.dev #"cuda" if torch.cuda.is_available() else "cpu"
num_workers = args.workers
rs = args.thres

# # Parameters
# loss_threshold = float("-inf")
# qp = 32
# batch_size = 1  # No paper 32
# device = 0 #"cuda" if torch.cuda.is_available() else "cpu"
# num_workers = 2
# thres = 0.45


print("Using {} device".format(device))



# Paths
#l_path_val = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels/mod_with_real_CTU/complexity/mod_with_struct_change_eval_32x32/"#/balanced_labels_downsamp"  # For evaluation 

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
                utils.echo("Complete:{percentage:.0%}".format(percentage=i_batch / size))

            # Delete later condition
            # if i_batch == 10:
            #    break

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

    # Print information about training
    #print()
    #print("Validation in validation data:")
    #print(classification_report(ground_truths, predictions))
    #print()

    ## Compute metrics from validation
    #f1_test, recall_test, precision_test, accuracy_test =\
    #        train_model_utils.model_simple_metrics(predictions, ground_truths)

    ## Compute confusion matrix
    # conf_mat_test = train_model_utils.compute_conf_matrix(predictions, ground_truths)
    ## Compute ROC curve
    # ROC_test = train_model_utils.compute_ROC_curve(pred_vector, ground_truths_vector, predictions)
    
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
    file_name = "results_stg_6_thres_"+str(rs)
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
    stg1_2 = MSECNN.MseCnnStg_1_v2(device=device, QP=qp).to(device)
    stg3 = MSECNN.MseCnnStg_x_v2(device=device, QP=qp).to(device)
    stg4 = MSECNN.MseCnnStg_x_v2(device=device, QP=qp).to(device)
    stg5 = MSECNN.MseCnnStg_x_v2(device=device, QP=qp).to(device)
    stg6 = MSECNN.MseCnnStg_x_v2(device=device, QP=qp).to(device)
    model = (stg1_2, stg3, stg4, stg5, stg6)

    ans = 'y'#str(input('Do you want to load any existing model? Y/N \n'))
    if ans == 'Y' or ans == 'y':
       #path = ""
       # path = input('What\'s the path of the files you want to load the model to?')
       model = train_model_utils.load_model_parameters_eval(model, path, device)

    # Prepare testing data, Dataset and Dataloader
    val_data = CustomDataset.CUDatasetStg6ComplV5(files_path=l_path_val)
    batch_sampler_val = CustomDataset.SamplerStg6(val_data, batch_size)  # Batch Sampler
    dataloader_val = DataLoader(val_data, num_workers=num_workers, batch_sampler=batch_sampler_val)

    # Compute CU proportions
    #print("Computing CU proportions...")
    #pm = torch.reshape(torch.tensor([0.5, 0.5, 0.00001, 0.00001, 0.00001, 0.00001]), shape=(1, -1)).to(device) # torch.reshape(torch.tensor([0.5, 0.5, 0.00001, 0.00001, 0.00001, 0.00001]), shape=(1, -1))  #torch.reshape(torch.tensor([0.70, 0.01, 0.01, 0.01, 0.01, 0.01]), shape=(1, -1)) #torch.reshape(torch.tensor([0.0009995002498750624, 0.999000499750125, 0.0000000000000000001, 0.0000000000000000001, 0.0000000000000000001, 0.0000000000000000001]), shape=(1, -1)) # Delete later #torch.reshape(torch.tensor([0.5, 0.5, 0.00001, 0.00001, 0.00001, 0.00001]), shape=(1, -1)).to(device) #torch.reshape(torch.tensor([0.0001, 0.999, 0.00001, 0.00001, 0.00001, 0.00001]), shape=(1, -1)) # torch.reshape(torch.tensor([0.5, 0.5, 0.00001, 0.00001, 0.00001, 0.00001]), shape=(1, -1))  #torch.reshape(torch.tensor([0.70, 0.01, 0.01, 0.01, 0.01, 0.01]), shape=(1, -1)) #torch.reshape(torch.tensor([0.0009995002498750624, 0.999000499750125, 0.0000000000000000001, 0.0000000000000000001, 0.0000000000000000001, 0.0000000000000000001]), shape=(1, -1)) # Delete later
    #pm, am = dataset_utils.compute_split_proportions_with_custom_data_multi_new(train_data, -2)
    #pm = torch.reshape(torch.tensor(list(pm.values()))+0.000000000000001, shape=(1, -1)).to(device)

    # Train
    print("Starting training...")
    val_setup(dataloader_val, model, device)


if __name__ == "__main__":
    main()

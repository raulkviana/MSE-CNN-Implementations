"""@package docstring 

@file train_stg6.py 

@brief Training script for the sixth stage of the MSE-CNN, for the luma channel.  
 
@section libraries_train_stg6 Libraries 
- sklearn.metrics
- MSECNN
- torch.utils.data
- torch
- argparse
- torch.utils.tensorboard
- datetime
- train_model_utils
- utils
- numpy
- constants
- CustomDataset
- sys
- time
- matplotlib.pyplot

@section classes_train_stg6 Classes 
- None

@section functions_train_stg6 Functions 
- train(dataloader, model, loss_fn, optimizer, device)
- test(dataloader, model, loss_fn, device, loss_name)
- train_test(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, epochs, lr_sch)
- main()
 
@section global_vars_train_stg6 Global Variables 
- learning_rate
- parser 
- args 
- loss_threshold 
- batch_size 
- qp 
- device 
- n_mod 
- num_workers 
- writer 
- l_path_val 
- decay
- decay_controler
- iterations
- files_mod_name_stats 
- l_path_train
- l_path_test
- cnt_train
- cnt_test_train
- cnt_test_test

@section todo_train_stg6 TODO 
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

@section author_train_stg6 Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:23:10.689038
"""


# Imports
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import time
import argparse
from torch.utils.tensorboard import SummaryWriter 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import CustomDataset
    import train_model_utils
    import MSECNN
    import constants
    import utils
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")


# Parse arguments
parser = argparse.ArgumentParser(description=constants.script_description)
parser.add_argument("-i", type=int)
parser.add_argument("--batch", type=int)
parser.add_argument("--dev", type=int)
parser.add_argument("-b", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("--workers", type=int)
parser.add_argument("--nmod", type=str)
parser.add_argument("--dcontr", type=int)

# Get arguments
args = parser.parse_args()

# Parameters
beta = args.b
learning_rate = args.lr
loss_threshold = float("-inf")
QP = 32
batch_size = args.batch  # No paper 32
iterations = args.i  # No paper 500_000
decay = 0.01
decay_controler = args.dcontr
device = args.dev #"cuda" if torch.cuda.is_available() else "cpu"
num_workers = args.workers
n_mod = args.nmod

print("Using {} device".format(device))

# Tensorboard variable
writer = SummaryWriter("runs/MSECNN_"+n_mod)

# Paths
# Data
l_path_train = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_stg6_v4/train_valid_test/balanced_labels_downsamp/train_val"  # For training Labels path
l_path_test = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_stg6_v4/train_valid_test/balanced_labels_downsamp/test"  # For testing Labels path


# Build name modifier
files_mod_name_stats = "_multi_batch_iter_{ite}_batch_{batch}_QP_{QP}_beta_{be}_lr_{lr}_{n_mod}".format(ite=iterations, batch=batch_size, QP=QP, be=beta, lr=learning_rate, n_mod=n_mod)

# Register for loss counter
cnt_train = 0
cnt_test_train = 0
cnt_test_test = 0

def train(dataloader, model, loss_fn, optimizer, device):
    """!
    If batch size equal to 1 it's a Stochastic Gradiente Descent (SGD), otherwise it's a mini-batch gradient descent. 
    If the batch is the same as the number as the size of the dataset, it will be a Batch gradient Descent
    """

    # Initialize variable
    size = len(dataloader.dataset)
    size = int(size/batch_size)
    global cnt_train

    # History variables for loss
    loss_RD_lst = []
    loss_CE_lst = []
    loss_lst = []

    model[0].eval()
    model[1].eval()
    model[2].eval()
    model[3].eval()
    model[4].train()

    # Loop
    for batch_num, sample_batch in enumerate(dataloader):
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

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Convert type
        CUs = CTU.to(device)
        Y = split.to(device)
        RDs = RDs.to(dtype=torch.float64).to(device)
        Y = train_model_utils.one_hot_enc(torch.tensor(Y.tolist())).to(device)

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

        # Compute the loss and its gradients
        loss, loss_CE, loss_RD = loss_fn(pred, Y, RDs)
        loss.backward()

        # Register the losses
        loss_lst.append(loss.item())
        loss_CE_lst.append(loss_CE.item())
        loss_RD_lst.append(loss_RD.item())
        cnt_train += 1

        # Adjust learning weights
        optimizer.step()

        # Print information about training
        if (batch_num+1) % 1 == 0:

            #Acc_history.append(acc)
            utils.echo("Complete: {percentage:.0%}".format(percentage=batch_num/size))

        # Save model
        if (batch_num+1) % 5 == 0:

            f_name = "last_stg_"  # File name
            for k in range(len(model)):
                train_model_utils.save_model_parameters("stg6_best_"+files_mod_name_stats, f_name + str(k), model[k])

    # Get mean of losses
    mean_L_RD = np.array(loss_RD_lst).mean()
    mean_L_CE = np.array(loss_CE_lst).mean()
    mean_L = np.array(loss_lst).mean()

    # Register losses per epoch
    writer.add_scalars("Losses/trainPerEpoch", {"Loss": mean_L, "Loss_CE": mean_L_CE, 
                        "Loss_RD": mean_L_RD}, t)

def test(dataloader, model, loss_fn, device, loss_name):
    # Initialize variables
    size = len(dataloader.dataset)
    size = int(size/batch_size)
    # History variables
    predictions = []
    ground_truths = []
    pred_vector = []
    ground_truths_vector = []
    loss_RD_lst = []
    loss_CE_lst = []
    loss_lst = []
    
    global cnt_test_test
    global cnt_test_train

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

            # Convert type
            CUs = CTU.to(device)
            Y = split.to(device)
            RDs = RDs.to(dtype=torch.float64).to(device)
            Y = train_model_utils.one_hot_enc(torch.tensor(Y.tolist())).to(device)

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

            # Compute loss
            loss, loss_CE, loss_RD = loss_fn(pred, Y, RDs)

            if loss_name == "train":
                loss_lst.append(loss.item())
                loss_CE_lst.append(loss_CE.item())
                loss_RD_lst.append(loss_RD.item())
                cnt_test_train += 1
            
            else:
                # Register losses
                loss_lst.append(loss.item())
                loss_CE_lst.append(loss_CE.item())
                loss_RD_lst.append(loss_RD.item())
                cnt_test_test += 1

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
    
    # Get mean of losses
    mean_L_RD = np.array(loss_RD_lst).mean()
    mean_L_CE = np.array(loss_CE_lst).mean()
    mean_L = np.array(loss_lst).mean()

    # Register losses per epoch
    writer.add_scalars("Losses/val"+loss_name+"PerEpoch", {"Loss": mean_L, "Loss_CE": mean_L_CE, 
                        "Loss_RD": mean_L_RD}, t)

    return predictions, ground_truths, pred_vector, ground_truths_vector

def train_test(train_dataloader, test_dataloader, model, loss_fn, optimizer, device, epochs, lr_sch):
    # Print current time
    train_model_utils.print_current_time()
    
    # History variables
    # Train
    predictions_train = []
    ground_truths_train = []
    # Test
    predictions_test = []
    ground_truths_test = []

    # Initialize best f1-score register
    best_f1 = -1
    
    # Start training
    global t
    for t in range(epochs):
        
        # Cycle start time
        start = time.time()

        # Print current epoch
        print(f"Epoch {t + 1}\n-------------------------------")

        # Train
        print("TRAINING")
        train(train_dataloader, model, loss_fn, optimizer, device)

        print()

        # Validation
        print("VALIDATING")
        # Validate on training data
        predictions_train, ground_truths_train, pred_vector_train, gt_vector_train\
            = test(train_dataloader, model, loss_fn, device, "train")

        # Validate on validation data
        predictions_test, ground_truths_test, pred_vector_test, gt_vector_test\
            = test(test_dataloader, model, loss_fn, device, "test")

        # Print information about training
        print()
        print("Validation in training data:")
        print(classification_report(ground_truths_train, predictions_train))
        print()
        print("Validation in validation data:")
        print(classification_report(ground_truths_test, predictions_test))
        print()

        # Compute metrics from training
        f1_train, recall_train, precision_train, accuracy_train = \
            train_model_utils.model_simple_metrics(predictions_train, ground_truths_train)
        # Compute top k accuracy
        top_2_accuracy_train = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector_train), torch.tensor(ground_truths_train), 2)
        top_3_accuracy_train = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector_train), torch.tensor(ground_truths_train), 3)
        top_4_accuracy_train = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector_train), torch.tensor(ground_truths_train), 4)
        # Compute multi-thresholdin performance
        pred_mult_thres_0_3_train = utils.multi_thresholding(0.3, torch.tensor(pred_vector_train))
        mul_thre_perfo_0_3_train = train_model_utils.compute_multi_thres_performance(pred_mult_thres_0_3_train, ground_truths_train)
        pred_mult_thres_0_5_train = utils.multi_thresholding(0.5, torch.tensor(pred_vector_train))
        mul_thre_perfo_0_5_train = train_model_utils.compute_multi_thres_performance(pred_mult_thres_0_5_train, ground_truths_train)
        writer.add_scalars("metrics/train", {"f1-score": f1_train, "recall": recall_train, 
                          "precision": precision_train, "accuracy": accuracy_train, "top 2 accuracy": top_2_accuracy_train,
                          "top 3 accuracy": top_3_accuracy_train, "top 4 accuracy": top_4_accuracy_train, "multi-thres 0.3": mul_thre_perfo_0_3_train, "multi-thres 0.5": mul_thre_perfo_0_5_train, }, t)
        
        # Compute ROC curve
        ROC_train = train_model_utils.compute_ROC_curve(pred_vector_train, gt_vector_train, predictions_train)
        # ROC curve: Training data
        writer.add_figure("rocCurve/train", ROC_train, t)
        plt.clf()
        # Compute confusion matrix
        conf_mat_train = train_model_utils.compute_conf_matrix(predictions_train, ground_truths_train)
        # Confusion matrix: Training data
        writer.add_figure("confMatrix/train", conf_mat_train, t)

        # Compute metrics from validation
        f1_test, recall_test, precision_test, accuracy_test =\
             train_model_utils.model_simple_metrics(predictions_test, ground_truths_test)
        # Compute top k accuracy
        top_2_accuracy_test = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector_test), torch.tensor(ground_truths_test), 2)
        top_3_accuracy_test = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector_test), torch.tensor(ground_truths_test), 3)
        top_4_accuracy_test = train_model_utils.compute_top_k_accuracy(torch.tensor(pred_vector_test), torch.tensor(ground_truths_test), 4)
        # Compute multi-thresholdin performance
        pred_mult_thres_0_3_test = utils.multi_thresholding(0.3, torch.tensor(pred_vector_test))
        mul_thre_perfo_0_3_test = train_model_utils.compute_multi_thres_performance(pred_mult_thres_0_3_test, ground_truths_test)
        pred_mult_thres_0_5_test = utils.multi_thresholding(0.5, torch.tensor(pred_vector_test))
        mul_thre_perfo_0_5_test = train_model_utils.compute_multi_thres_performance(pred_mult_thres_0_5_test, ground_truths_test)
        # Save metric from training
        writer.add_scalars("metrics/val", {"f1-score": f1_test, "recall": recall_test, 
                          "precision": precision_test, "accuracy": accuracy_test, "top 2 accuracy": top_2_accuracy_test,
                          "top 3 accuracy": top_3_accuracy_test, "top 4 accuracy": top_4_accuracy_test, "multi-thres 0.3": mul_thre_perfo_0_3_test, "multi-thres 0.5": mul_thre_perfo_0_5_test}, t)

        # Compute ROC curve
        ROC_test = train_model_utils.compute_ROC_curve(pred_vector_test, gt_vector_test, predictions_test)
        # ROC curve: Validation data
        writer.add_figure("rocCurve/test", ROC_test, t)
        plt.clf()
        # Compute confusion matrix
        conf_mat_test = train_model_utils.compute_conf_matrix(predictions_test, ground_truths_test)
        # Confusion matrix: Validation data
        writer.add_figure("confMatrix/val", conf_mat_test, t)

        # Save best loss and best params
        if best_f1 < f1_test:
            best_f1 = f1_test  # Update best f1-score
            f_name = "best_stg_"  # Choose file name
            for k in range(len(model)):
                train_model_utils.save_model_parameters("stg6_last_"+files_mod_name_stats, f_name + str(k), model[k])

        # Learning rate Decay
        if (t+1) % decay_controler == 0:
            print("lr:", lr_sch.get_lr())
            lr_sch.step()
        
        end = time.time()
        print("Time:", str(datetime.timedelta(seconds=end-start)))

    # Print current time
    train_model_utils.print_current_time()
    print("Done!")

    # Make sure all events have been written to the disk
    writer.flush()
    writer.close() # Close session

    return model, optimizer

def main():

    # Initialize Model
    stg1_2 = MSECNN.MseCnnStg1(device=device, QP=32).to(device)
    stg3 = MSECNN.MseCnnStgX(device=device, QP=32).to(device)
    stg4 = MSECNN.MseCnnStgX(device=device, QP=32).to(device)
    stg5 = MSECNN.MseCnnStgX(device=device, QP=32).to(device)
    stg6 = MSECNN.MseCnnStgX(device=device, QP=32).to(device)

    model = (stg1_2, stg3, stg4, stg5, stg6)

    # Load Optimizer
    optimizer = torch.optim.Adam(model[-1].parameters(), lr=learning_rate)  # Tal como no paper

    # Load scheduler
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay, last_epoch=-1)

    ans = str(input('Do you want to load any existing model? Y/N \n'))
    if ans == 'Y' or ans == 'y':

        ans = str(input('Do you want a full model (all stages in the folder)? Y/N \n'))
        path = input('What\'s the path of the files you want to load the model to?')

        if ans == 'Y' or ans == 'y':
            model = train_model_utils.load_model_parameters_eval(model, path, device)
            
        else:
            model = train_model_utils.load_model_stg_5_stg_6(model, path, device)

    # Prepare training and testing data, Dataset and Dataloader
    train_data = CustomDataset.CUDatasetStg6(files_path=l_path_train) 
    batch_sampler_train = CustomDataset.SamplerStg6(train_data, batch_size)  # Batch Sampler
    dataloader_train = DataLoader(train_data, num_workers=num_workers, batch_sampler=batch_sampler_train)
    test_data = CustomDataset.CUDatasetStg6(files_path=l_path_test)
    batch_sampler_test = CustomDataset.SamplerStg6(test_data, batch_size)  # Batch Sampler
    dataloader_test = DataLoader(test_data, num_workers=num_workers, batch_sampler=batch_sampler_test)
    
    # Load Loss function
    loss_fn = MSECNN.LossFunctionMSE(beta=beta)

    # Train
    print("Starting training...")
    my_final_model, my_final_optimizer = train_test(dataloader_train, dataloader_test, model, loss_fn,
                                                    optimizer, device, iterations,
                                                    lr_sch)

    # Print optimizer's state_dict
    print("Optimizer's lr:", lr_sch.get_lr())
    print("Optimizer's lr:", lr_sch.get_last_lr())
    print("Optimizer's lr:", my_final_optimizer.param_groups[0]["lr"])

if __name__ == "__main__":
    main()

"""@package docstring 

@file train_model_utils.py 

@brief Group of functions that are supposed to be used directly in the training or evaluation scripts
 
@section libraries_train_model_utils Libraries 
- os
- torch
- numpy
- matplotlib.pyplot
- dataset_utils
- seaborn
- itertools
- datetime
- sklearn.metrics

@section classes_train_model_utils Classes 
- None
 
@section functions_train_model_utils Functions 
- model_statistics(J_history, predicted, ground_truth, pred_vector, gt_vector,
- compute_conf_matrix(predicted, ground_truth)
- compute_top_k_accuracy(pred_vector, gt_vector, topk)
- compute_num_splits_sent(pred_lst)
- compute_multi_thres_performance(pred_lst, gt_lst)
- compute_ROC_curve(pred_vector, gt_vector, pred_num)
- model_simple_metrics(predicted, ground_truth)
- obtain_best_modes(rs, pred)
- obtain_mode(pred)
- one_hot_enc(tensor, num_classes=6)
- print_parameters(model, optimizer)
- save_model_parameters(dir_name, f_name, model)
- save_model(dir_name, f_name, model, optimizer, loss, acc)
- load_model_parameters_stg(model, path, stg, dev)
- load_model_parameters_eval(model, path, dev)
- load_model_stg_12_stg_3(model, path, dev)
- load_model_stg_3_stg_4(model, path, dev)
- load_model_stg_4_stg_5(model, path, dev)
- load_model_stg_5_stg_6(model, path, dev)
- print_current_time()
 
@section global_vars_train_model_utils Global Variables 
- None

@section todo_train_model_utils TODO 
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

@section author_train_model_utils Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:22:04.154941
"""

# ==============================================================
# Imports
# ==============================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import dataset_utils
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, \
    precision_score, roc_curve, auc
import seaborn as sns
from itertools import cycle

# ==============================================================
# Functions
# ==============================================================

def model_statistics(J_history, predicted, ground_truth, pred_vector, gt_vector,
                        f1_list, recall_list, precision_list,
                        accuracy_list, train_or_val="train"):
    """!
    @brief Evaluates model with metrics, such as accuracy and f1_score. This version plots the
           evolution of the metrics: f1-score, recall, precision, accuracy.

    @param [in] J_history: Loss function values over iterations
    @param [in] predicted: List of predictions made by the model with single value
    @param [in] ground_truth: List of the ground-truths with single value
    @param [in] pred_vector: List of predictions made by the model with vectors values
    @param [in] gt_vector: List of the ground-truths with vectors values
    @param [in] train_or_val: String that is used to write on the image files names
    @param [out] f1: F1 score
    @param [out] recall: Recall score
    @param [out] precision: Precision score
    @param [out] accuracy: Accuracy score
    """

    # Report
    report = classification_report(ground_truth, predicted)
    try:
        f = open("classification_report_" + train_or_val + ".txt", "x")
    except:
        os.remove("classification_report_" + train_or_val + ".txt")
        f = open("classification_report_" + train_or_val + ".txt", "x")

    f.write(report)
    f.close()

    # Plot Loss function throughout iterations
    plt.figure()
    plt.plot(J_history, label='loss')
    plt.xlabel("Batch number")
    plt.title("Loss: " + train_or_val)
    plt.legend()
    plt.grid()
    name = "Loss_" + train_or_val + ".png"
    plt.savefig(name)
    plt.clf()

    # Plot confusion matrix
    labels = ["Non-Split", "QT", "HBT", "VBT", "HTT", "VTT"]

    # Draw confusion matrix
    sns.heatmap(confusion_matrix(ground_truth, predicted), annot=True, fmt='d', cmap='Blues', yticklabels=labels,
                xticklabels=labels)
    name = "confusion_matrix_" + train_or_val + ".png"
    plt.title("Confusion Matrix: " + train_or_val)
    plt.savefig(name)
    plt.clf()

    # Plot metrics
    plt.figure()
    plt.plot(f1_list, 'go-', label='F1-score')
    plt.plot(recall_list, 'bo-', label='Recall')
    plt.plot(accuracy_list,'ro-',label='Accuracy')
    plt.plot(precision_list, 'yo-',label='Precision')
    plt.grid()
    plt.legend()
    plt.xlabel("Epochs")
    plt.title("Metrics: " + train_or_val)
    name = "metrics_evolution" + train_or_val + ".png"
    plt.savefig(name)
    plt.clf()

    # ROC Curves
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Convert to numpy
    y, y_score = np.array(gt_vector), np.array(pred_vector)

    # Number of classes
    n_classes = np.array(predicted).max() + 1

    # Obtain ROC curve values and area
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # # Plot all ROC curves
    plt.figure()
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.grid()
    name = "ROC_curve" + train_or_val + ".png"
    plt.savefig(name)
    plt.clf()

def compute_conf_matrix(predicted, ground_truth):
    """!
    @brief Computes the confusion matrix

    @param [in] predicted: List of predictions made by the model with single value
    @param [in] ground_truth: List of the ground-truths with single value
    @param [out] accuracy: Accuracy score
    """
    # Plot confusion matrix
    labels = ["Non-Split", "QT", "HBT", "VBT", "HTT", "VTT"]

    # Draw confusion matrix
    c_ma = confusion_matrix(ground_truth, predicted)

    # Obtain amount of examples for each class
    amount_lst = []
    for k in range(len(c_ma)):
        
        amount_lst.append(c_ma[k].sum())

    map = sns.heatmap(c_ma / np.array([amount_lst]).T, annot=True, fmt='.2f', cmap='Blues', yticklabels=labels,
                                        xticklabels=labels)

    map.set(xlabel="Predicted", ylabel="Ground-truth")

    return map.get_figure()

def compute_top_k_accuracy(pred_vector, gt_vector, topk):
    """!
    @brief Computes the top k accuracy score

    @param [in] predicted: List of predictions made by the model with probabilities for each split (pytorch tensor)
    @param [in] ground_truth: List of the ground-truths with single value (pytorch tensor)
    @param [in] topk: Number of best accuricies to choose
    @param [out] accuracy: Accuracy score
    """
    # Initialize variables
    n_entries = gt_vector.shape[0]
    res = 0

    # Get top predictions
    top_pred = torch.topk(pred_vector, topk)
    idxs = top_pred[1]  # Indexs

    for n in range(n_entries):

        if gt_vector[n] in idxs[n]:
            res += 1

    # Compute accuracy
    res = res / n_entries  
        
    return res

def compute_num_splits_sent(pred_lst):
    """!
    @brief Computes the num of splits that would be analyzed by the encoder

    @param [in] predicted: List of predictions made by the model with probabilities values
    @param [out] res: Mean of number of splits sent
    """
    # Initialize variables
    n_entries = len(pred_lst)
    
    temp  = [len(p) for p in pred_lst]
    res = sum(temp)/n_entries
        
    return res

def compute_multi_thres_performance(pred_lst, gt_lst):
    """!
    @brief Computes multi-threshold performance

    @param [in] predicted: List of predictions made by the model with integer value
    @param [in] ground_truth: List of the ground-truths with single value 
    @param [out] res: Accuracy score
    """
    # Initialize variables
    n_entries = len(gt_lst)
    
    temp  = [True for l, v in zip(pred_lst, gt_lst) if v in l]  # If the ground truth is within the predictions, add 1 to the list
    res = len(temp)

    # Compute accuracy
    res = res / n_entries  
        
    return res

def compute_ROC_curve(pred_vector, gt_vector, pred_num):
    """!
    @brief Computes ROC curve

    @param [in] pred_vector: List of predictions vectors (one-hot encoded)
    @param [in] gt_vector: List of the ground-truths vectors (one-hot encoded)
    @param [in] pred_num: List of the predicitons with numbers corresponding to partitions
    @return [out] figure: Figure with the ROC curve
    """
    # ROC Curves
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Convert to numpy
    y, y_score = np.array(gt_vector), np.array(pred_vector)

    # Number of classes
    n_classes = np.array(pred_num).max() + 1

    # Obtain ROC curve values and area
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # # Plot all ROC curves
    figure = plt.figure()
    lw = 2
    colors = cycle(["b", "y", "k", "r", "m", "g"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.grid()

    return figure

def model_simple_metrics(predicted, ground_truth):
    """!
    @brief Evaluates model with metrics 4 metrics, such as accuracy, f1_score, recall and precision

    @param [in] predicted: List of predictions made by the model with single value
    @param [in] ground_truth: List of the ground-truths with single value
    @param [out] f1: F1 score
    @param [out] recall: Recall score
    @param [out] precision: Precision score
    @param [out] accuracy: Accuracy score
    """

    # Compute metrics
    accuracy = accuracy_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted, average='weighted')
    recall = recall_score(ground_truth, predicted, average='weighted')
    precision = precision_score(ground_truth, predicted, average='weighted')

    return f1, recall, precision, accuracy

def obtain_best_modes(rs, pred):
    """!
    @brief Converts a prediction into a specific number that corresponds to the best way to split (non-split, quad tree,
           binary vert tree...)

    @param [in] rs: Thresholds
    @param [in] pred: Predicted values from the model with one-hot encoding
    @param [out] mode: Predicted values with the number of the mode
    """

    # Obtain maximum value for each prediction
    y_max = torch.reshape(torch.max(pred, dim=1)[0], shape=(-1, 1))
    # Obtain lowest possible value for each prediction
    y_max_rs = y_max * rs

    # Obtain logic truth where preds>=y_max_rs
    search_RD_logic = pred >= y_max_rs

    # Obtain the indexs that are true
    search_RD = torch.nonzero(torch.squeeze(search_RD_logic.int()), as_tuple=False)
    
    return search_RD

def obtain_mode(pred):
    """!
    @brief Converts a prediction into a specific number that corresponds to the best way to split (non-split, quad tree,
           binary vert tree...)

    @param [in] pred: Predicted values from the model with one-hot encoding
    @param [out] mode: Predicted values with the number of the mode
    """

    mode = torch.argmax(pred, dim=1)

    return mode

def one_hot_enc(tensor, num_classes=6):
    """!
    @brief Implements one-hot encoding to a specific tensor with the set of split modes

    @param [in] tensor: Tensor with a set of split modes
    @param [in] num_classes: Number classes in the tensor
    @param [out] new_tensor: Tensor with one-hot encoding implemented
    """

    new_tensor = torch.reshape(torch.nn.functional.one_hot(tensor, num_classes=num_classes), shape=(-1, num_classes))

    return new_tensor

def print_parameters(model, optimizer):
    """!
    @brief Prints the parameters from the state dictionaries of the model and optimizer

    @param [in] model: Model that the parameters will be printed
    @param [in] optimizer: Optimizer that the parameters will be printed
    """

    print()
    print()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print()

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    print()
    print()


def save_model_parameters(dir_name, f_name, model):
    """!
    @brief Saves only the model parameters to a specific folder

    @param [in] dir_name: Name of the directory where the parameters will be saved
    @param [in] f_name: Name of the file that the parameters will be saved on
    @param [in] model: Model which the parameters will be saved
    """

    # Check to see if directory already exists
    CURR_DIR = os.getcwd()  # Get current dir
    try:
        os.mkdir(CURR_DIR + '/' + dir_name)

    except:
        pass
    
    file_path = dir_name + '/' + f_name + '.pth'
    torch.save(model.state_dict(), file_path)
    print("Saved PyTorch Model State to", file_path)


def save_model(dir_name, f_name, model, optimizer, loss, acc):
    """!
    @brief Saves the parameters of the model and of the optimizer, and also the loss and the accuracy. These are saved
           into the folder specified by the user.

    @param [in] dir_name: Name of the directory where the parameters will be saved
    @param [in] f_name: Name of the file that the parameters will be saved on
    @param [in] model: Model which the parameters will be saved
    @param [in] optimizer: Optimizer which the parameters will be saved
    @param [in] loss: Loss value
    @param [in] acc: Accuracy value
    """

    # Check to see if directory already exists
    curr_dir = os.getcwd()  # Get current dir
    try:
        os.mkdir(curr_dir + '/' + dir_name)

    except:
        pass

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc
    }, dir_name + '/' + f_name + '.tar')


def load_model_parameters_stg(model, path, stg, dev):
    """!
    @brief Loads all stages but make sure that the stage number 'stg' has the same parameters has the previous

    @param [in] model: Model which the parameters will be loaded
    @param [in] path: Path/Folder containing the files that are supposed to be loaded
    @param [in] stg: Integer containing the last stage number to load
    @param [in] dev: Device do load the model to
    @param [out] model: Model loaded tuple
    """

    # Get files names
    files = dataset_utils.get_files_from_folder(path, endswith=".pth")
    files.sort()

    # Load state dict to each model
    for i in range(len(files)):
        # Load
        file_path = path + '/' + files[i]

        if type(dev) is int:
            m = torch.load(file_path, map_location="cuda:"+str(dev))
            print("Loading model to GPU number", dev)

        else:
            m = torch.load(file_path, map_location="cpu")
            print("Loading model to CPU")
        
        model[i].load_state_dict(m)

        # To avoid inconsistent inference results
        model[i].eval()

    # Specific stage must have the same parameters of the previous stage to it
    file_path = path + '/' + files[stg-2]

    if type(dev) is int:
        m = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        m = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")
    
    model[stg-1].load_state_dict(m)

    # To avoid inconsistent inference results
    model[stg-1].eval()

    return model

def load_model_parameters_eval(model, path, dev):
    """!
    @brief Loads all stages, meant to be used with the eval_model script

    @param [in] model: Model which the parameters will be loaded
    @param [in] path: Path/Folder containing the files that are supposed to be loaded
    @param [in] dev: Device do load the model to
    @param [out] model: Model loaded tuple
    """

    # Get files names
    files = dataset_utils.get_files_from_folder(path, endswith=".pth")
    files.sort()

    # Load state dict to each model
    for i in range(len(files)):
        # Load
        file_path = path + '/' + files[i]

        if type(dev) is int:
            m = torch.load(file_path, map_location="cuda:"+str(dev))
            print("Loading model to GPU number", dev)

        else:
            m = torch.load(file_path, map_location="cpu")
            print("Loading model to CPU")
        
        model[i].load_state_dict(m)

        # To avoid inconsistent inference results
        model[i].eval()

    return model

def load_model_stg_12_stg_3(model, path, dev):
    """!
    @brief THis function makes it possible to load parameters from the first and second stage to the third

    @param [in] model: Model which the parameters will be loaded, with 2 models (one for the first and second stage, and another for the third stage)
    @param [in] path: Path/Folder containing the files that are supposed to be loaded
    @param [in] dev: Device to load the model to
    """

    # Get files names
    files = dataset_utils.get_files_from_folder(path, endswith=".pth")
    files.sort()
    
    # Load state dict to each model
    file_path = path + "/" + files[0]
    stg_1_2 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_1_2 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_1_2 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[0].load_state_dict(stg_1_2)

    # Load specific layers
    with torch.no_grad():

        # Conditional convolution
        model[1].simple_conv[0].weight.copy_(stg_1_2["simple_conv_stg2.0.weight"])
        model[1].simple_conv[0].bias.copy_(stg_1_2["simple_conv_stg2.0.bias"])
        model[1].simple_conv[1].weight.copy_(stg_1_2["simple_conv_stg2.1.weight"])

        model[1].simple_conv_no_activation[0].weight.copy_(stg_1_2["simple_conv_no_activation_stg2.0.weight"])
        model[1].simple_conv_no_activation[0].bias.copy_(stg_1_2["simple_conv_no_activation_stg2.0.bias"])

        model[1].simple_conv2[0].weight.copy_(stg_1_2["simple_conv2_stg2.0.weight"])
        model[1].simple_conv2[0].bias.copy_(stg_1_2["simple_conv2_stg2.0.bias"])
        model[1].simple_conv2[1].weight.copy_(stg_1_2["simple_conv2_stg2.1.weight"])

        model[1].simple_conv_no_activation2[0].weight.copy_(stg_1_2["simple_conv_no_activation2_stg2.0.weight"])
        model[1].simple_conv_no_activation2[0].bias.copy_(stg_1_2["simple_conv_no_activation2_stg2.0.bias"])
        
        model[1].activation_PRelu.weight.copy_(stg_1_2["activation_PRelu_stg2.weight"])
        model[1].activation_PRelu2.weight.copy_(stg_1_2["activation_PRelu2_stg2.weight"])        

    # To avoid inconsistent inference results
    for m in model:

        m.eval()

    return model

def load_model_stg_3_stg_4(model, path, dev):
    """!
    @brief This function makes it possible to load parameters from the third stage to the fourth

    @param [in] model: Model which the parameters will be loaded, with 2 models (one for the first and second stage, and another for the third stage)
    @param [in] path: Path/Folder containing the files that are supposed to be loaded
    @param [in] dev: Device to load the model to
    """

    # Get files names
    files = dataset_utils.get_files_from_folder(path, endswith=".pth")

    # Load Stage 1 and 2
    for k in range(len(files)):

        if '0' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]
    stg_1_2 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_1_2 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_1_2 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[0].load_state_dict(stg_1_2)

    # Load Stage 3
    for k in range(len(files)):

        if '1' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]

    stg_3 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_3 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_3 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[-2].load_state_dict(stg_3)

    model[-1].load_state_dict(stg_3)

    # To avoid inconsistent inference results
    for m in model:

        m.eval()

    return model


def load_model_stg_4_stg_5(model, path, dev):
    """!
    @brief This function makes it possible to load parameters from the fourth stage to the fith

    @param [in] model: Model which the parameters will be loaded, with 2 models (one for the first and second stage, and another for the third stage)
    @param [in] path: Path/Folder containing the files that are supposed to be loaded
    @param [in] dev: Device to load the model to
    """

    # Get files names
    files = dataset_utils.get_files_from_folder(path, endswith=".pth")

    # Load Stage 1 and 2
    for k in range(len(files)):
        if '0' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]
    stg_1_2 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_1_2 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_1_2 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[0].load_state_dict(stg_1_2)

    # Load Stage 3
    for k in range(len(files)):
        if '1' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]

    stg_3 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_3 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_3 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[1].load_state_dict(stg_3)


    # Load Stage 4
    for k in range(len(files)):

        if '2' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]

    stg_4 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_4 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_4 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[2].load_state_dict(stg_4)

    # Load stage 5
    model[3].load_state_dict(stg_4)

    # To avoid inconsistent inference results
    for m in model:
        m.eval()

    return model


def load_model_stg_5_stg_6(model, path, dev):
    """!
    @brief This function makes it possible to load parameters from the fourth stage to the fith

    @param [in] model: Model which the parameters will be loaded, with 2 models (one for the first and second stage, and another for the third stage)
    @param [in] path: Path/Folder containing the files that are supposed to be loaded
    @param [in] dev: Device to load the model to
    """

    # Get files names
    files = dataset_utils.get_files_from_folder(path, endswith=".pth")

    # Load Stage 1 and 2
    for k in range(len(files)):
        if '0' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]
    stg_1_2 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_1_2 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_1_2 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[0].load_state_dict(stg_1_2)

    # Load Stage 3
    for k in range(len(files)):
        if '1' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]

    stg_3 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_3 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_3 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[1].load_state_dict(stg_3)


    # Load Stage 4
    for k in range(len(files)):
        if '2' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]

    stg_4 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_4 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_4 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[2].load_state_dict(stg_4)

    # Load Stage 5
    for k in range(len(files)):
        if '3' in files[k]:
            break
    
    # Load state dict to each model
    file_path = path + "/" + files[k]

    stg_5 = torch.load(file_path, map_location="cuda:"+str(dev))
    if type(dev) is int:
        stg_5 = torch.load(file_path, map_location="cuda:"+str(dev))
        print("Loading model to GPU number", dev)

    else:
        stg_5 = torch.load(file_path, map_location="cpu")
        print("Loading model to CPU")

    model[3].load_state_dict(stg_5)

    # Load stage 6
    model[4].load_state_dict(stg_5)

    # To avoid inconsistent inference results
    for m in model:
        m.eval()

    return model

def print_current_time():
    """!
    @brief Prints current time
    """
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
 



    
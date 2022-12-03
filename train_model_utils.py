"""@package docstring 

@file train_model_utils.py 

@brief Group of functions that are supposed to be used directly in the training or evaluation scripts
 
@section libraries_train_model_utils Libraries 
- os
- torch
- numpy
- dataset_utils
- matplotlib.pyplot
- MSECNN
- itertools
- sklearn.metrics
- seaborn
- datetime
- math

@section classes_train_model_utils Classes 
- None 

@section functions_train_model_utils Functions 
- def split(in_CU, split_mode)
- model_statistics(J_history, predicted, ground_truth, pred_vector, gt_vector, train_or_val="train")
- model_statistics_v2(J_history, predicted, ground_truth, pred_vector, gt_vector,
- compute_conf_matrix(predicted, ground_truth)
- right_size(CUs)
- compute_top_k_accuracy(pred_vector, gt_vector, topk)
- compute_num_splits_sent(pred_lst)
- compute_multi_thres_performance(pred_lst, gt_lst)
- compute_ROC_curve(pred_vector, gt_vector, pred_num)
- model_simple_metrics(predicted, ground_truth)
- obtain_best_modes(rs, pred)
- obtain_mode(pred)
- one_hot_enc(tensor, num_classes=6)
- split_class(in_CU, split_mode)
- split_class_multi_batch(in_CU, split_mode)
- split_class_multi_batch_v2_for_testing(in_CU, split_mode, pos)
- split_class_multi_batch_v2(in_CU, split_mode)
- nr_calc(ac, ap)
- find_right_cu_class(fm, cu_pos)
- find_right_cu_class_multi_batch(CUs, cus_pos)
- def split(in_CU, split_mode)
- print_parameters(model, optimizer)
- save_model_parameters(dir_name, f_name, model)
- save_model(dir_name, f_name, model, optimizer, loss, acc)
- load_model_parameters_stg(model, path, stg, dev)
- load_model_parameters_eval(model, path, dev)
- load_model_parameters(model, path, dev)
- load_model(model, optimizer, path, dev)
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
- Last time modified is 2022-12-02 18:21:21.208167
"""

# Imports
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import MSECNN
import dataset_utils
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, \
    precision_score, roc_curve, auc
import seaborn as sns
from itertools import cycle

def model_statistics(J_history, predicted, ground_truth, pred_vector, gt_vector, train_or_val="train"):
    """!
    @brief Evaluates model with metrics, such as accuracy and f1_score

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

    # Compute metrics
    accuracy = accuracy_score(ground_truth, predicted)
    f1 = f1_score(ground_truth, predicted, average='micro')
    recall = recall_score(ground_truth, predicted, average='micro')
    precision = precision_score(ground_truth, predicted, average='micro')
    #print("precision", precision, "f1", f1, "accuracy", accuracy, "recall", recall)

    # Report
    report = classification_report(ground_truth, predicted)
    try:
        f = open("classification_report_" + train_or_val + ".txt", "x")
    except:
        os.remove("classification_report_" + train_or_val + ".txt")
        f = open("classification_report_" + train_or_val + ".txt", "x")

    f.write(report)
    f.close()

    # Total amount of data
    total_data = len(ground_truth)

    # Plot Loss function throughout iterations
    plt.figure()
    plt.plot(J_history, label='loss')
    plt.xlabel("Batch number")
    plt.title("Loss: " + train_or_val)
    plt.legend()
    plt.grid()
    name = "Loss_" + train_or_val + ".png"
    plt.savefig(name)
    #plt.show()
    plt.clf()

    # Plot confusion matrix
    labels = ["Non-Split", "QT", "HBT", "VBT", "HTT", "VTT"]

    # Draw confusion matrix
    sns.heatmap(confusion_matrix(ground_truth/total_data, predicted/total_data), annot=True, fmt='d', cmap='Blues', yticklabels=labels,
                xticklabels=labels)
    #disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(ground_truth, predicted), display_labels=labels[0:2])
    #disp.plot()
    name = "confusion_matrix_" + train_or_val + ".png"
    plt.title("Confusion Matrix: " + train_or_val)
    plt.savefig(name)
    #plt.show()
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
    # plt.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    #     color="deeppink",
    #     linestyle=":",
    #     linewidth=4,
    # )

    # plt.plot(
    #     fpr["macro"],
    #     tpr["macro"],
    #     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    #     color="navy",
    #     linestyle=":",
    #     linewidth=4,
    # )

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
    #plt.show()
    name = "ROC_curve" + train_or_val + ".png"
    plt.savefig(name)
    plt.clf()

    return f1, recall, precision, accuracy

def model_statistics_v2(J_history, predicted, ground_truth, pred_vector, gt_vector,
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
    #plt.show()
    plt.clf()

    # Plot confusion matrix
    labels = ["Non-Split", "QT", "HBT", "VBT", "HTT", "VTT"]

    # Draw confusion matrix
    sns.heatmap(confusion_matrix(ground_truth, predicted), annot=True, fmt='d', cmap='Blues', yticklabels=labels,
                xticklabels=labels)
    #disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(ground_truth, predicted), display_labels=labels[0:2])
    #disp.plot()
    name = "confusion_matrix_" + train_or_val + ".png"
    plt.title("Confusion Matrix: " + train_or_val)
    plt.savefig(name)
    #plt.show()
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
    #plt.show()
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
    #plt.show()
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

def right_size(CUs):
    """!
        @brief Verify if the CU as the right size: height as to be lower than width

        @param [in] CUs: Feature maps
        @param [out] Boolean value indicating the right size
    """
    
    return False if CUs.shape[-2] > CUs.shape[-1] else True

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
        @brief Computes the num of splits that would be sent to the encoder

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
        @brief Computes the confusion matrix

        @param [in] predicted: List of predictions made by the model with single value
        @param [in] ground_truth: List of the ground-truths with single value
        @param [out] accuracy: Accuracy score
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

def split_class(in_CU, split_mode):
    """!
    @brief Split a CU in one of the specific modes (quad tree, binary vert tree, binary horz tree, threenary vert tree,
           etc). It does the same as the split fucntion but it expects the inserted CU to be of the class CU

    @param [in] in_CU: CU (class) inserted in the network, e.g. (1, 1, 128, 128)
    @param [in] split_mode: Most likely split mode, in scalar value (0, 1, 2, 3, 4, or 5)
    @param [out] out_CU: Output of tuple with the CUs
    """

    # Initialize variable
    out_CUs = []
    CU_h = in_CU.f_maps.size()[-2]
    CU_w = in_CU.f_maps.size()[-1]

    if split_mode == 0:  # Non-split
        out_CUs.append(in_CU)

    elif split_mode == 1:  # Quad tree
        # Split in four equal size tensors
        temp, temp2 = torch.split(in_CU.f_maps, (int(CU_h / 2), int(CU_h / 2)), 2)
        temp_1, temp_2 = torch.split(temp, (int(CU_w / 2), int(CU_w / 2)), 3)
        temp2_1, temp2_2 = torch.split(temp2, (int(CU_w / 2), int(CU_w / 2)), 3)

        # Create new CUs
        position1 = in_CU.position
        new_CU1 = MSECNN.CU(position1, temp_1, in_CU.ap)

        position2 = {'CU_loc_left': in_CU.position['CU_loc_left'] + int(CU_w / 2),
                     'CU_loc_top': in_CU.position['CU_loc_top']}
        new_CU2 = MSECNN.CU(position2, temp_2, in_CU.ap)

        position3 = {'CU_loc_left': in_CU.position['CU_loc_left'],
                     'CU_loc_top': in_CU.position['CU_loc_top'] + int(CU_h / 2)}
        new_CU3 = MSECNN.CU(position3, temp2_1, in_CU.ap)

        position4 = {'CU_loc_left': in_CU.position['CU_loc_left'] + int(CU_w / 2),
                     'CU_loc_top': in_CU.position['CU_loc_top'] + int(CU_h / 2)}
        new_CU4 = MSECNN.CU(position4, temp2_2, in_CU.ap)

        # Merge all
        out_CUs.extend([new_CU1, new_CU2, new_CU3, new_CU4])

    elif split_mode == 2:  # Binary Horz tree
        # Split in 2 vertical equal size tensors
        temp1, temp2 = torch.split(in_CU.f_maps, (int(CU_h / 2), int(CU_h / 2)), 2)

        # Create new CUs
        position1 = in_CU.position
        new_CU1 = MSECNN.CU(position1, temp1, in_CU.ap)

        position2 = {'CU_loc_left': in_CU.position['CU_loc_left'],
                     'CU_loc_top': in_CU.position['CU_loc_top'] + int(CU_h / 2)}
        new_CU2 = MSECNN.CU(position2, temp2, in_CU.ap)

        # Merge all
        out_CUs.extend([new_CU1, new_CU2])

    elif split_mode == 3:  # Binary Vert tree
        # Split in 2 Horz equal size tensors
        temp1, temp2 = torch.split(in_CU.f_maps, (int(CU_w / 2), int(CU_w / 2)), 3)

        # Create new CUs
        position1 = in_CU.position
        new_CU1 = MSECNN.CU(position1, temp1, in_CU.ap)

        position2 = {'CU_loc_left': in_CU.position['CU_loc_left'] + int(CU_w / 2),
                     'CU_loc_top': in_CU.position['CU_loc_top']}
        new_CU2 = MSECNN.CU(position2, temp2, in_CU.ap)

        # Merge all
        out_CUs.extend([new_CU1, new_CU2])

    elif split_mode == 4:  # Ternary Horz tree
        # Split in 3 parts with the ratio of 1,2,1
        temp_1, temp_2, temp_3 = torch.split(in_CU.f_maps, (int(CU_h / 4), int(CU_h / 2), int(CU_h / 4)), 2)

        # Create new CUs
        position1 = in_CU.position
        new_CU1 = MSECNN.CU(position1, temp_1, in_CU.ap)

        position2 = {'CU_loc_left': in_CU.position['CU_loc_left'],
                     'CU_loc_top': in_CU.position['CU_loc_top'] + int(CU_h / 4)}
        new_CU2 = MSECNN.CU(position2, temp_2, in_CU.ap)

        position3 = {'CU_loc_left': in_CU.position['CU_loc_left'],
                     'CU_loc_top': in_CU.position['CU_loc_top'] + int(3 * CU_h / 4)}
        new_CU3 = MSECNN.CU(position3, temp_3, in_CU.ap)

        # Merge all
        out_CUs.extend([new_CU1, new_CU2, new_CU3])

    elif split_mode == 5:  # Ternary Vert tree
        # Split in 3 parts with the ratio of 1,2,1
        temp_1, temp_2, temp_3 = torch.split(in_CU.f_maps, (int(CU_w / 4), int(CU_w / 2), int(CU_w / 4)), 3)

        # Create new CUs
        position1 = in_CU.position
        new_CU1 = MSECNN.CU(position1, temp_1, in_CU.ap)

        position2 = {'CU_loc_left': in_CU.position['CU_loc_left'] + int(CU_w / 4),
                     'CU_loc_top': in_CU.position['CU_loc_top']}
        new_CU2 = MSECNN.CU(position2, temp_2, in_CU.ap)

        position3 = {'CU_loc_left': in_CU.position['CU_loc_left'] + int(3 * CU_w / 4),
                     'CU_loc_top': in_CU.position['CU_loc_top']}
        new_CU3 = MSECNN.CU(position3, temp_3, in_CU.ap)

        # Merge all
        out_CUs.extend([new_CU1, new_CU2, new_CU3])

    return out_CUs

def split_class_multi_batch(in_CU, split_mode):
    """!
    @brief Split a CU in one of the specific modes (quad tree, binary vert tree, binary horz tree, threenary vert tree,
           etc). It does the same as the split fucntion but it expects the inserted CU to be of the class CU

    @param [in] in_CU: CU (class) inserted in the network, e.g. (1, 1, 128, 128)
    @param [in] split_mode: List with the most likely split mode, in scalar value (0, 1, 2, 3, 4, or 5)
    @param [out] out_CU: Output of tuple with the CUs
    """

    # Initialize variable
    CU_h = in_CU.f_maps.size()[-2]
    CU_w = in_CU.f_maps.size()[-1]
    list_cus = []
    list_pos = []
    list_aps = []

    for k in range(in_CU.f_maps.size()[0]):

        if split_mode[k] == 0:  # Non-split
            list_cus.append(in_CU.f_maps_list[k])
            list_pos.append(in_CU.position_list[k])
            list_aps.append(in_CU.ap_list[k])

        elif split_mode[k] == 1:  # Quad tree
            # Split in four equal size tensors
            temp, temp2 = torch.split(torch.unsqueeze(in_CU.f_maps[k, :, :, :], 0), (int(CU_h / 2), int(CU_h / 2)), 2)
            temp_1, temp_2 = torch.split(temp, (int(CU_w / 2), int(CU_w / 2)), 3)
            temp2_1, temp2_2 = torch.split(temp2, (int(CU_w / 2), int(CU_w / 2)), 3)

            # Create new CUs
            position1 = in_CU.position_list[k]
            list_cus.append(temp_1)
            list_pos.append(position1)
            list_aps.append(in_CU.ap_list[k])

            position2 = (in_CU.position_list[k][0], in_CU.position_list[k][1] + int(CU_w / 2))  # top, left
            list_cus.append(temp_2)
            list_pos.append(position2)
            list_aps.append(in_CU.ap_list[k])

            position3 = (in_CU.position_list[k][0] + int(CU_h / 2), in_CU.position_list[k][1])  # top, left
            list_cus.append(temp2_1)
            list_pos.append(position3)
            list_aps.append(in_CU.ap_list[k])

            position4 = (in_CU.position_list[k][0] + int(CU_h / 2), in_CU.position_list[k][1] + int(CU_w / 2))  # top, left
            list_cus.append(temp2_2)
            list_pos.append(position4)
            list_aps.append(in_CU.ap_list[k])

        elif split_mode[k] == 2:  # Binary Horz tree
            # Split in 2 vertical equal size tensors
            temp1, temp2 = torch.split(torch.unsqueeze(in_CU.f_maps[k, :, :, :], 0), (int(CU_h / 2), int(CU_h / 2)), 2)

            # Create new CUs
            position1 = in_CU.position_list[k]
            list_cus.append(temp1)
            list_pos.append(position1)
            list_aps.append(in_CU.ap_list[k])

            position2 = (in_CU.position_list[k][0] + int(CU_h / 2), in_CU.position_list[k][1])  # top, left
            list_cus.append(temp2)
            list_pos.append(position2)
            list_aps.append(in_CU.ap_list[k])

        elif split_mode[k] == 3:  # Binary Vert tree
            # Split in 2 Horz equal size tensors
            temp1, temp2 = torch.split(torch.unsqueeze(in_CU.f_maps[k, :, :, :], 0), (int(CU_w / 2), int(CU_w / 2)), 3)

            # Create new CUs
            position1 = in_CU.position_list[k]
            list_cus.append(temp1)
            list_pos.append(position1)
            list_aps.append(in_CU.ap_list[k])

            position2 = (in_CU.position_list[k][0] + int(CU_h / 2), in_CU.position_list[k][1] + int(CU_w / 2))  # top, left
            list_cus.append(temp2)
            list_pos.append(position2)
            list_aps.append(in_CU.ap_list[k])

        elif split_mode[k] == 4:  # Ternary Horz tree
            # Split in 3 parts with the ratio of 1,2,1
            temp_1, temp_2, temp_3 = torch.split(torch.unsqueeze(in_CU.f_maps[k, :, :, :], 0), (int(CU_h / 4), int(CU_h / 2), int(CU_h / 4)), 2)

            # Create new CUs
            position1 = in_CU.position_list[k]
            list_cus.append(temp_1)
            list_pos.append(position1)
            list_aps.append(in_CU.ap_list[k])

            position2 = (in_CU.position_list[k][0] + int(CU_h / 4), in_CU.position_list[k][1])
            list_cus.append(temp_2)
            list_pos.append(position2)
            list_aps.append(in_CU.ap_list[k])

            position3 = (in_CU.position_list[k][0] + int(3 * CU_h / 4), in_CU.position_list[k][1])
            list_cus.append(temp_3)
            list_pos.append(position3)
            list_aps.append(in_CU.ap_list[k])

        elif split_mode[k] == 5:  # Ternary Vert tree
            # Split in 3 parts with the ratio of 1,2,1
            temp_1, temp_2, temp_3 = torch.split(torch.unsqueeze(in_CU.f_maps[k, :, :, :], 0), (int(CU_w / 4), int(CU_w / 2), int(CU_w / 4)), 3)

            # Create new CUs
            position1 = in_CU.position_list[k]
            list_cus.append(temp_1)
            list_pos.append(position1)
            list_aps.append(in_CU.ap_list[k])

            position2 = (in_CU.position_list[k][0], in_CU.position_list[k][1] + int(CU_w / 4))
            list_cus.append(temp_2)
            list_pos.append(position2)
            list_aps.append(in_CU.ap_list[k])

            position3 = (in_CU.position_list[k][0], in_CU.position_list[k][1] + int(3 * CU_w / 4))
            list_cus.append(temp_3)
            list_pos.append(position3)
            list_aps.append(in_CU.ap_list[k])

    # Create new structure
    out_CUs = MSECNN.CU_batch(list_pos, list_cus, list_aps)

    return out_CUs

def split_class_multi_batch_v2_for_testing(in_CU, split_mode, pos):
    """!
    @brief Split a CU in one of the specific modes (quad tree, binary vert tree, binary horz tree, threenary vert tree,
           etc). It does the same as the split fucntion but it expects the inserted CU to be of the class CU

    @param [in] in_CU: CU (class) inserted in the network, e.g. (1, 1, 128, 128)
    @param [out] out_CU: Output of tuple with the CUs
    """

    # Initialize variable
    CU_h = in_CU.f_maps.size()[-2]
    CU_w = in_CU.f_maps.size()[-1]
    list_cus = []
    list_pos = []

    # Modify variable
    #if pos.shape[0] == 1:
    #    pos = pos[0]

    #else:
    #    pos = torch.unsqueeze(pos, 0)

    if split_mode == 0:  # Non-split
        list_cus.append(in_CU.f_maps)
        list_pos.append(pos)

    elif split_mode == 1:  # Quad tree
        # Split in four equal size tensors
        temp, temp2 = torch.split(in_CU.f_maps, (int(CU_h / 2), int(CU_h / 2)), 2)
        temp_1, temp_2 = torch.split(temp, (int(CU_w / 2), int(CU_w / 2)), 3)
        temp2_1, temp2_2 = torch.split(temp2, (int(CU_w / 2), int(CU_w / 2)), 3)

        # Create new CUs
        position1 = pos
        list_cus.append(temp_1)
        list_pos.append(position1)

        position2 = (pos[:, 0], pos[:, 1] + int(CU_w / 2))  # top, left
        list_cus.append(temp_2)
        list_pos.append(position2)

        position3 = (pos[:, 0] + int(CU_h / 2), pos[:, 1])  # top, left
        list_cus.append(temp2_1)
        list_pos.append(position3)

        position4 = (pos[:, 0] + int(CU_h / 2), pos[:, 1] + int(CU_w / 2))  # top, left
        list_cus.append(temp2_2)
        list_pos.append(position4)

    elif split_mode == 2:  # Binary Horz tree
        # Split in 2 vertical equal size tensors
        temp1, temp2 = torch.split(in_CU.f_maps, (int(CU_h / 2), int(CU_h / 2)), 2)

        # Create new CUs
        position1 = pos
        list_cus.append(temp1)
        list_pos.append(position1)

        position2 = (pos[:, 0] + int(CU_h / 2), pos[:, 1])  # top, left
        list_cus.append(temp2)
        list_pos.append(position2)

    elif split_mode == 3:  # Binary Vert tree
        # Split in 2 Horz equal size tensors
        temp1, temp2 = torch.split(in_CU.f_maps, (int(CU_w / 2), int(CU_w / 2)), 3)

        # Create new CUs
        position1 = pos
        list_cus.append(temp1)
        list_pos.append(position1)

        position2 = (pos[:, 0], pos[:, 1] + int(CU_w / 2))  # top, left
        list_cus.append(temp2)
        list_pos.append(position2)

    elif split_mode == 4:  # Ternary Horz tree
        # Split in 3 parts with the ratio of 1,2,1
        temp_1, temp_2, temp_3 = torch.split(in_CU.f_maps, (int(CU_h / 4), int(CU_h / 2), int(CU_h / 4)), 2)

        # Create new CUs
        position1 = pos
        list_cus.append(temp_1)
        list_pos.append(position1)

        position2 = (pos[:, 0] + int(CU_h / 4), pos[:, 1])
        list_cus.append(temp_2)
        list_pos.append(position2)

        position3 = (pos[:, 0] + int(3 * CU_h / 4), pos[:, 1])
        list_cus.append(temp_3)
        list_pos.append(position3)

    elif split_mode == 5:  # Ternary Vert tree
        # Split in 3 parts with the ratio of 1,2,1
        temp_1, temp_2, temp_3 = torch.split(in_CU.f_maps, (int(CU_w / 4), int(CU_w / 2), int(CU_w / 4)), 3)

        # Create new CUs
        position1 = pos
        list_cus.append(temp_1)
        list_pos.append(position1)

        position2 = (pos[:, 0], pos[:, 1] + int(CU_w / 4))
        list_cus.append(temp_2)
        list_pos.append(position2)

        position3 = (pos[:, 0], pos[:, 1] + int(3 * CU_w / 4))
        list_cus.append(temp_3)
        list_pos.append(position3)

    # Concatenate CUs
    CUs = torch.cat(tuple(list_cus), dim=0)

    # Create new structure
    out_CUs = MSECNN.CU_batch_v2(CUs)

    return out_CUs, list_pos


def split_class_multi_batch_v2(in_CU, split_mode):
    """!
    @brief Split a CU in one of the specific modes (quad tree, binary vert tree, binary horz tree, threenary vert tree,
           etc). It does the same as the split fucntion but it expects the inserted CU to be of the class CU

    @param [in] in_CU: CU (class) inserted in the network, e.g. (1, 1, 128, 128)
    @param [out] out_CU: Output of tuple with the CUs
    """

    # Initialize variable
    CU_h = in_CU.f_maps.size()[-2]
    CU_w = in_CU.f_maps.size()[-1]
    list_cus = []

    if split_mode == 0:  # Non-split
        list_cus.append(in_CU.f_maps)

    elif split_mode == 1:  # Quad tree
        # Split in four equal size tensors
        temp, temp2 = torch.split(in_CU.f_maps, (int(CU_h / 2), int(CU_h / 2)), 2)
        temp_1, temp_2 = torch.split(temp, (int(CU_w / 2), int(CU_w / 2)), 3)
        temp2_1, temp2_2 = torch.split(temp2, (int(CU_w / 2), int(CU_w / 2)), 3)

        # Create new CUs
        list_cus.append(temp_1)
        list_cus.append(temp_2)
        list_cus.append(temp2_1)
        list_cus.append(temp2_2)

    elif split_mode == 2:  # Binary Horz tree
        # Split in 2 vertical equal size tensors
        temp1, temp2 = torch.split(in_CU.f_maps, (int(CU_h / 2), int(CU_h / 2)), 2)

        # Create new CUs
        list_cus.append(temp1)
        list_cus.append(temp2)

    elif split_mode == 3:  # Binary Vert tree
        # Split in 2 Horz equal size tensors
        temp1, temp2 = torch.split(in_CU.f_maps, (int(CU_w / 2), int(CU_w / 2)), 3)

        # Create new CUs
        list_cus.append(temp1)
        list_cus.append(temp2)

    elif split_mode == 4:  # Ternary Horz tree
        # Split in 3 parts with the ratio of 1,2,1
        temp_1, temp_2, temp_3 = torch.split(in_CU.f_maps, (int(CU_h / 4), int(CU_h / 2), int(CU_h / 4)), 2)

        # Create new CUs
        list_cus.append(temp_1)
        list_cus.append(temp_2)
        list_cus.append(temp_3)

    elif split_mode == 5:  # Ternary Vert tree
        # Split in 3 parts with the ratio of 1,2,1
        temp_1, temp_2, temp_3 = torch.split(in_CU.f_maps, (int(CU_w / 4), int(CU_w / 2), int(CU_w / 4)), 3)

        # Create new CUs
        list_cus.append(temp_1)
        list_cus.append(temp_2)
        list_cus.append(temp_3)

    # Concatenate CUs
    CUs = torch.cat(tuple(list_cus), dim=0)

    # Create new structure
    out_CUs = MSECNN.CU_batch_v2(CUs)

    return out_CUs

def nr_calc(ac, ap):
    """!
    @brief Calculate the number of residual units

    @param [in] ac: Minimum value of the current input axises
    @param [in] ap: Minimum value of the parent input axises
    @param [out] nr: Number of residual units
    """

    nr = 0

    if (ac != 0):

        if (ac == 128):
            nr = 1

        elif (ap != 0):

            if (4 <= ac <= 64):
                nr = int(math.log2(ap / ac))

            else:
                raise Exception("ac with invalid number! ac =", ac)

        else:
            raise Exception("ap with invalid number! ap =", ap)

    else:
        raise Exception("ac with invalid number! ac =", ac)

    return nr


def find_right_cu_class(fm, cu_pos):
    """!
    @brief Find the respective CU given the position of the CU

    @param [in] fm: Feature maps with various CUs/Feature Maps
    @param [in] cu_pos: Position to find the CU
    @param [out] f: Feature map that corresponds to the CU
    """

    for f in fm:

        if f.position["CU_loc_left"] == cu_pos["CU_loc_left"] and f.position["CU_loc_top"] == cu_pos["CU_loc_top"]:
            return f

    raise Exception("CU not found! This should not happen. Can not find position "+str(cu_pos) + " in " + str(fm))

def find_right_cu_class_multi_batch(CUs, cus_pos):
    """!
    @brief Find the respective CU given the position of the CU

    @param [in] fm: Feature maps with various CUs/Feature Maps
    @param [in] cus_pos: Positions of the CUs
    @param [out] f: Feature map that corresponds to the CU
    """
    list_cus = []
    list_pos = []
    list_aps = []

    for k in range(len(cus_pos)):

        for i in range(len(CUs.position_list)):

            if cus_pos[k] == CUs.position_list[i]:
                list_cus.append(CUs.f_maps_list[i])
                list_pos.append(CUs.position_list[i])
                list_aps.append(CUs.ap_list[i])
                break

    if len(cus_pos) != len(list_cus):
        raise Exception("CU not found! This should not happen. Can not find position " + str(cus_pos) + " in " + str(CUs))

    return MSECNN.CU_batch(list_pos, list_cus, list_aps)


# def split(in_CU, split_mode):
#     """
#     @brief Split a CU in one of the specific modes (quad tree, binary vert tree, binary horz tree, threenary vert tree, etc)
#
#     @param [in] in_CU: CU inserted in the network, e.g. (1, 1, 128, 128)
#     @param [in] split_mode: Most likely split mode, in scalar value (0, 1, 2, 3, 4, or 5)
#     @param [out] out_CU: Output of tuple with the CUs
#     """
#
#     # Initialize variable
#     out_CU = torch.tensor([[]])
#     CU_h = in_CU.cu.size()[-1]
#     CU_w = in_CU.cu.size()[-2]
#
#     if (split_mode == 0):  # Non-split
#         out_CU = in_CU
#
#     elif (split_mode == 1):  # Quad tree
#         # Split in four equal size tensors
#         temp, temp2 = torch.split(in_CU.cu, (int(CU_w / 2), int(CU_w / 2)), 2)
#         temp_1, temp_2 = torch.split(temp, (int(CU_h / 2), int(CU_h / 2)), 3)
#         temp_3, temp_4 = torch.split(temp2, (int(CU_h / 2), int(CU_h / 2)), 3)
#
#         # Create new CUs
#         position1 = in_CU.position
#         new_CU1 = MSECNN.CUInstance(position1, temp_1, in_CU.ap)
#
#         position2 = (in_CU.position[0], in_CU.position[1]+int(CU_h / 2))
#         new_CU2 = MSECNN.CUInstance(position2, temp_2, in_CU.ap)
#
#         position3 = (in_CU.position[0]+int(CU_w / 2), in_CU.position[1])
#         new_CU3 = MSECNN.CUInstance(position3, temp_3, in_CU.ap)
#
#         position4 = (in_CU.position[0]+int(CU_w / 2), in_CU.position[1]+int(CU_h / 2))
#         new_CU4 = MSECNN.CUInstance(position4, temp_4, in_CU.ap)
#
#         # Merge all
#         out_CU = (new_CU1, new_CU2, new_CU3, new_CU4)
#
#     elif (split_mode == 2):  # Binary Vert tree
#         # Split in 2 vertical equal size tensors
#         temp1, temp2 = torch.split(in_CU.cu, (int(CU_w / 2), int(CU_w / 2)), 3)
#
#         # Create new CUs
#         position1 = in_CU.position
#         new_CU1 = MSECNN.CUInstance(position1, temp1, in_CU.ap)
#
#         position2 = (in_CU.position[0] + int(CU_w / 2), in_CU.position[1])
#         new_CU2 = MSECNN.CUInstance(position2, temp2, in_CU.ap)
#
#         # Merge all
#         out_CU = (new_CU1, new_CU2)
#
#     elif (split_mode == 3):  # Binary Horz tree
#         # Split in 2 Horz equal size tensors
#         temp1, temp2 = torch.split(in_CU.cu, (int(CU_h / 2), int(CU_h / 2)), 2)
#
#         # Create new CUs
#         position1 = in_CU.position
#         new_CU1 = MSECNN.CUInstance(position1, temp1, in_CU.ap)
#
#         position2 = (in_CU.position[0], in_CU.position[1] + int(CU_h / 2))
#         new_CU2 = MSECNN.CUInstance(position2, temp2, in_CU.ap)
#
#         # Merge all
#         out_CU = (new_CU1, new_CU2)
#
#
#     elif (split_mode == 4):  # Ternary Vert tree # Cortar 3 pedaços em que o maior tem um ratio de 1:1 para a soma dos outros 2 pedaços (Ou seja, divide a imagem em 8 pedaços e depois agrupas 4 pedaços, 2 pedeços e mais 2 pedaços. Cada conjunto desses é uma divisão). Por exemeplo: 2,4,2
#         # Split in 3 parts with the ratio of 1,2,1
#         temp_1, temp_2, temp_3 = torch.split(in_CU.cu, (int(CU_w / 4), int(CU_w / 2), int(CU_w / 4)), 2)
#
#         # Create new CUs
#         position1 = in_CU.position
#         new_CU1 = MSECNN.CUInstance(position1, temp_1, in_CU.ap)
#
#         position2 = (in_CU.position[0] + int(CU_w / 4), in_CU.position[1])
#         new_CU2 = MSECNN.CUInstance(position2, temp_2, in_CU.ap)
#
#         position3 = (in_CU.position[0] + int(CU_w / 2), in_CU.position[1])
#         new_CU3 = MSECNN.CUInstance(position3, temp_3, in_CU.ap)
#
#         # Merge all
#         out_CU = (new_CU1, new_CU2, new_CU3)
#
#
#     elif (split_mode == 5):  # Ternary Horz tree
#         # Split in 3 parts with the ratio of 1,2,1
#         temp_1, temp_2, temp_3 = torch.split(in_CU.cu, (int(CU_h / 4), int(CU_h / 2), int(CU_h / 4)), 3)
#
#         # Create new CUs
#         position1 = in_CU.position
#         new_CU1 = MSECNN.CUInstance(position1, temp_1, in_CU.ap)
#
#         position2 = (in_CU.position[0], in_CU.position[1] + int(CU_h / 4))
#         new_CU2 = MSECNN.CUInstance(position2, temp_2, in_CU.ap)
#
#         position3 = (in_CU.position[0], in_CU.position[1] + int(CU_w / 2))
#         new_CU3 = MSECNN.CUInstance(position3, temp_3, in_CU.ap)
#
#         # Merge all
#         out_CU = (new_CU1, new_CU2, new_CU3)
#
#     return out_CU

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

    torch.save(model.state_dict(), dir_name + '/' + f_name + '.pth')
    #print("Saved PyTorch Model State to", f_name + ".pth")


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

    # torch.save(model.state_dict(), dir_name+'/' + f_name + '.pth')

    #print("Saved PyTorch Model State to", f_name + ".tar")

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

def load_model_parameters(model, path, dev):
    """!
    @brief Loads only the model parameters from a specified files, but it considers that if the model has more stages
           than files with the parameters, the difference of the amount is 1, this means that it's expected that the
           last stage in the model variable will have the same parameters has the penultimate.

    @param [in] model: Model which the parameters will be loaded
    @param [in] path: Path/Folder containing the files that are supposed to be loaded
    @param [in] dev: Device do load the model to
    @param [out] model: Model loaded tuple
    """

    # Get files names
    files = dataset_utils.get_files_from_folder(path, endswith=".pth")
    files.sort()

    # Guarante that the number of files is just one number below the number of models
    assert len(model) - len(files) == 1

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

    # Last new stage needs to have the same params as the stage before it
    if len(model) > len(files):
        file_path = path + '/' + files[-1]

        if type(dev) is int:
            m = torch.load(file_path, map_location="cuda:"+str(dev))
            print("Loading model to GPU number", dev)

        else:
            m = torch.load(file_path, map_location="cpu")
            print("Loading model to CPU")

        model[-1].load_state_dict(m)
        # To avoid inconsistent inference results
        model[-1].eval()

    return model

def load_model(model, optimizer, path, dev):
    """!
    @brief Loads the parameters of the model and of the optimizer. These are loaded from the folder specified by the
           user.

    @param [in] model: Model which the parameters will be loaded
    @param [in] optimizer: Optimizer which the parameters will be loaded
    @param [in] path: Path/Folder containing the files that are supposed to be loaded
    @param [in] dev: Device do load the model to
    """

    # Get files names
    files = dataset_utils.get_files_from_folder(path, endswith=".tar")
    files.sort()

    # Load state dict to each model
    for i in range(len(model)):
        # Load
        file_path = path + files[i]

        if type(dev) is int:
            checkpoint = torch.load(file_path, map_location="cuda:"+str(dev))
            print("Loading model to GPU number", dev)

        else:
            checkpoint = torch.load(file_path, map_location="cpu")
            print("Loading model to CPU")
         
        model[i].load_state_dict(checkpoint["model_state_dict"])
        optimizer[i].load_state_dict(checkpoint["optimizer_state_dict"])

        # To avoid inconsistent inference results
        model[i].eval()

    return model, optimizer

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
 



    
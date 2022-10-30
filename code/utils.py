# Imports
from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import math
import dataset_utils

def echo(string, padding=80):
    """
    @brief Prints and clears

    @param [in] string: String that will be printed
    @param [in] padding: Padding to avoid concatenations

    """
    padding = " " * (padding - len(string)) if padding else ""
    print(string + padding, end='\r')

def multi_thresholding(rs, preds):
    """
    Accordingly, the multi-threshold values can be chosen in the
    following strategies, ensuring the overall prediction accuracy
    of MSE-CNN.

    • Case 1 (more time saving): if the average threshold
    (1/5) * sum(rs) >= 0.4, then τ2 ≥ τ6 ≥ τ3 ≈ τ4 ≈ τ5.

    • Case 2 (better RD performance): if the average threshold
    (1/5) * sum(rs) < 0.4, then τ2 ≥ τ4 ≈ τ3 ≈ τ5 ≥ τ6.

    @brief Implementation of the multi-threshold for MSE-CNN

    @param [in] rs: Constant to control the minimum amount of thr probabilities
    @param [in] preds: Predictions made by the MSE-CNN
    @param [out] search_RD: Vector with the information of which modes to compute the RD cost and make a decision regarding in which split to make
    """

    # rs has to be between 0 and 1
    assert 0 < rs < 1

    # Obtain maximum value for each prediction
    y_max = torch.reshape( torch.max(preds, dim=1)[0], shape=(-1, 1))
    # Obtain lowest possible value for each prediction
    y_max_rs = y_max * rs

    # Obtain logic truth where preds>=y_max_rs
    search_RD_logic = preds >= y_max_rs

    # Obtain the indexs that are true
    search_RD = [[idx for idx, l2 in enumerate(l) if l2] for l in search_RD_logic.tolist()]

    return search_RD


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

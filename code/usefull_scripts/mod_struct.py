"""
Modifies structure to the JF recommendations one: [POC, pic_name, real_CTU, split_tens, RD_tens, pos_tens]
"""

import dataset_utils
import torch
import os
import time
import utils
import shutil
import pandas as pd
import numpy as np
import threading

# Main Function
def main():
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels/mod_with_real_CTU/"  # Path with the labels processed
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU"  # Path with the labels processed
    path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels/mod_with_real_CTU/complexity/test"
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels/mod_with_real_CTU/complexity"  # For training Labels path
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels_for_testing/test"  # Path with the labels processed
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_stg4_v4/train_valid_test/balanced_labels_downsamp/test"

    print("Modifying struct from:", path_dir_l)

    dataset_utils.change_struct_no_dupl_stg_6_complexity_v4(path_dir_l)
    #dataset_utils.change_struct_no_dupl_stg3_v4(path_dir_l)
    #dataset_utils.change_struct_8x8_no_dupl_v2(path_dir_l)
    #dataset_utils.change_struct_32x16_no_dupl_v2(path_dir_l)
    #dataset_utils.change_struct_no_dupl_stg5_v4(path_dir_l)
    #dataset_utils.change_struct_no_dupl_stg6_v4(path_dir_l)
    #dataset_utils.change_struct_32x32_no_dupl_v3(path_dir_l)
    #dataset_utils.change_struct_32x32(path_dir_l)
    #dataset_utils.change_struct_32x32_no_dupl_v2(path_dir_l)
    #dataset_utils.change_struct_64x64(path_dir_l)
    #dataset_utils.change_struct_32x32_eval(path_dir_l)

if __name__ == "__main__":
    main()
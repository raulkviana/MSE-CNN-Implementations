"""@package docstring 

@file balance_data.py 

@brief Balance the dataset through upsampling or downsampling.
 
@section libraries_balance_data Libraries 
- dataset_utils
- time

@section classes_balance_data Classes 
- None 

@section functions_balance_data Functions 
- main()
 
@section global_vars_balance_data Global Variables 
- None

@section todo_balance_data TODO 
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

@section author_balance_data Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2022-12-02 18:21:21.104253
"""

# ==============================================================
# Imports
# ==============================================================

import time
import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import dataset_utils
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")
# ==============================================================
# Main
# ==============================================================

def main():

    # Directory containing the .txt files with CUs informations
    path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_32x32_v2/train_valid_test"  # Path with the labels processed
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels_for_testing/train/mod_with_struct_change_32x32"
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_stg6_v4/train_valid_test/balanced_labels_downsamp/train_val"

    print("Balance data in:", path_dir_l)

    # start = time.time()
    # dataset_utils.balance_dataset_JF(path_dir_l, n_classes=6)  # Upsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_JF:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_down(path_dir_l, n_classes=2)  # Downsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_down:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_up(path_dir_l, n_classes=6)  # Upsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_up:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_down_v2(path_dir_l)  # Downsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_down_v2:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_down_v4(path_dir_l)  # Downsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_down_v4:", end-start)

    start = time.time()
    dataset_utils.balance_dataset_down_v3(path_dir_l)  # Downsampling
    end = time.time()
    print("Elapsed time for balance_dataset_down_v3:", end-start)

    # start = time.time()
    # pos_to_find_split = -3
    # dataset_utils.get_some_data_equaly(100000, path_dir_l, 2, pos_to_find_split)  # Downsampling
    # end = time.time()
    # print("Elapsed time for get_some_data_equaly:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_up_v2(path_dir_l)  # Upsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_up_v2:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_up_v3(path_dir_l)  # Upsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_up_v3:", end-start)

if __name__ == "__main__":
    main()

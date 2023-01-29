"""@package docstring 

@file mod_struct.py 

@brief Obtains structures following a specific guideline and without repetition of data 
 
@section libraries_mod_struct Libraries 
- dataset_utils
- sys
- Exception("Module

@section classes_mod_struct Classes 
- None 
@section functions_mod_struct Functions 
- main()
 
@section global_vars_mod_struct Global Variables 
- None 

@section todo_mod_struct TODO 
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

@section author_mod_struct Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:23:10.686048
"""


"""
Modifies structure to the JF recommendations one: [POC, pic_name, real_CTU, split_tens, RD_tens, pos_tens]
"""

import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import dataset_utils
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")

# Main Function
def main():
    path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels/mod_with_real_CTU/complexity/test"

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
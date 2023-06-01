"""@package docstring 

@file create_labels.py 

@brief Create data structures from raw data for luma channel. 
 
@section libraries_create_labels Libraries 
- dataset_utils
- sys

@section classes_create_labels Classes 
- None 
@section functions_create_labels Functions 
- main()
 
@section global_vars_create_labels Global Variables 
- None 

@section todo_create_labels TODO 
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

@section author_create_labels Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:23:10.672038
"""

import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import dataset_utils
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")


def main():

    # Directory containing the .dat files with CUs informations
    d_path = r""

    dataset_utils.unite_labels_v6_mod(dir_path_l=d_path, n_output_file="processed_labels")  # The result is saved in the same folder

if __name__ == "__main__":
    main()

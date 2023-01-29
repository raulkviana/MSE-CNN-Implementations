"""@package docstring 

@file get_cu_components.py 

@brief Obtains a specific CU and its specific components 
 
@section libraries_get_cu_components Libraries 
- dataset_utils
- sys
- Exception("Module
- CustomDataset

@section classes_get_cu_components Classes 
- None 
@section functions_get_cu_components Functions 
- main()
 
@section global_vars_get_cu_components Global Variables 
- None 

@section todo_get_cu_components TODO 
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

@section author_get_cu_components Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:23:10.684038
"""

import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import dataset_utils
    import CustomDataset
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")


def main():

    # Directory containing the .txt files with CUs informations
    f_path = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/data/RAISE_Test_768x512.yuv"
    file_info = dataset_utils.get_file_metadata_info("/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/data/", "RAISE_Test_768x512.yuv")
    f_size = (file_info["height"], file_info["width"])
    cu_pos = (0, 0)
    cu_size = (128, 128)
    frame_number = 0
    print("Path:", f_path)
    yuv_frame, CU_Y, CU_U, CU_V = CustomDataset.get_cu(f_path, f_size, cu_pos, cu_size, frame_number)

if __name__ == "__main__":
    main()

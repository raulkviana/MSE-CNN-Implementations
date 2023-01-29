"""@package docstring 

@file obtain_cch.py 

@brief Filters structured data and retrieves data in the luma or chroma channel 
 
@section libraries_obtain_cch Libraries 
- dataset_utils
- sys
- Exception("Module

@section classes_obtain_cch Classes 
- None 
@section functions_obtain_cch Functions 
- main()
 
@section global_vars_obtain_cch Global Variables 
- None 

@section todo_obtain_cch TODO 
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

@section author_obtain_cch Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:23:10.687041
"""


import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import dataset_utils
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")

def main():

    # Directory containing the .txt files with CUs informations
    d_path = r"C:\Users\Raul\Dropbox\Dataset\Img\Test_place\labels_pickle"

    dataset_utils.labels_with_specific_cch(dir_path=d_path, cch=0)
    dataset_utils.labels_with_specific_cch(dir_path=d_path, cch=1)

if __name__ == "__main__":
    main()

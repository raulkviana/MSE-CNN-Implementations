"""@package docstring 

@file encode_dataset.py 

@brief Encodes a .yuv files using the VTM-7.0 software 
 
@section libraries_encode_dataset Libraries 
- dataset_utils

@section classes_encode_dataset Classes 
- None 

@section functions_encode_dataset Functions 
- main()
 
@section global_vars_encode_dataset Global Variables 
- None 

@section todo_encode_dataset TODO 
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

@section author_encode_dataset Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2022-12-02 18:21:21.155854
"""

import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import dataset_utils
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")


def main():

    # Quantization parameter
    #QP = str(input("Qual o valor para o quantization parameter (QP)?\n"))
    QP = 27

    # Temporal Subsample Ratio
    #ts = str(input("Qual o valor para o Temporal Subsample Ratio (ts)?\n"))
    ts = 500

    # Directory containing the dataset images
    #d_path = str(input("Em que path absoluto está o dataset?\n"))
    d_path = '/mnt/c/Users/Raul/Dropbox/Dataset/Img/Example'
    d_path = "C:/Users/Raul/Dropbox/Dataset/Img/Example"

    # Directory containing the .exe file extension to run the encoder.
    # It ends with CPIV\VTM-7.0_Data\bin\vs16\msvc-19.24\x86_64\release in Windows
    e_path = "C:/Users/Raul/Documents/GitHub/CPIV/VTM-7.0_Data/bin/vs16/msvc-19.24/x86_64/release" # Example for windows
    #e_path = "/mnt/c/Users/Raul/Documents/GitHub/CPIV/VTM-7.0_Data/bin/" # Example for linux
    #e_path = str(input("Em que path absoluto está o encoder?\n"))

    dataset_utils.encode_dataset(d_path=d_path, e_path=e_path, ts=ts, QP=QP) # The result is saved in the encoder folder

if __name__ == "__main__":
    main()

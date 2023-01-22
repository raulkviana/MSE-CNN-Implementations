"""@package docstring 

@file add_CTU_v3.py 

@brief Adds actual CTUs to the data structure. This version is meant to be used with stage oriented data structure, but is similar to the previous version.
 
@section libraries_add_CTU_v3 Libraries 
- torch
- os
- shutil
- dataset_utils
- utils
- CustomDataset

@section classes_add_CTU_v3 Classes 
- None 

@section functions_add_CTU_v3 Functions 
- main()
 
@section global_vars_add_CTU_v3 Global Variables 
- None

@section todo_add_CTU_v3 TODO 
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

@section author_add_CTU_v3 Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2022-12-02 18:21:21.099308
"""

# ==============================================================
# Imports
# ==============================================================

import torch
import os
import shutil
import sys
# Insert the path of modules folder 
sys.path.insert(0, "../")
try:
    import dataset_utils
    import CustomDataset
    import utils
except:
    raise Exception("Module not found! Please verify that the main modules (CustomDataset, dataset_utils, MSECNN, train_model_utils and utils) can be found in the directory above the current one. Or just find a way of importing them.")

# ==============================================================
# Main
# ==============================================================

def main():
    # Path with the labels processed
    path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels"  
 
    # Path with the pictures
    path_dir_p = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/data"  

    # Create new dir to save data
    name_mod = "mod_with_real_CTU"
    new_dir = path_dir_l + "/" + name_mod + "/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    # CTU Counter and number of entries to accumulate
    ctu_count = 0

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        accum = data_size 

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):

            # Build all requirements to get the CU
            file_name = orig_list[k]['pic_name']
            img_path = os.path.join(path_dir_p, file_name + '.yuv')
            POC = orig_list[k]['POC']
            f_size = CustomDataset.get_file_size(file_name)  # Dict with the size of the image
            f_size = (f_size['height'], f_size['width'])
            cu_pos = (orig_list[k]["stg_1"]["cu_pos"]["CU_loc_top"], orig_list[k]["stg_1"]["cu_pos"]["CU_loc_left"])
            cu_size = (orig_list[k]["stg_1"]["cu_size"]["height"], orig_list[k]["stg_1"]["cu_size"]["width"])

            # Verify size of the CU
            if cu_size[0] != 128 or cu_size[1] != 128:
                continue

            # Get CU
            frame_CU, CU_Y, CU_U, CU_V = CustomDataset.get_cu(img_path, f_size, cu_pos, cu_size, POC)

            # Convert to Pytorch Tensor
            CU_Y = torch.tensor(CU_Y.tolist())

            # Convert to type
            CU_Y = CU_Y.to(dtype=torch.float32)

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Add to entry
            orig_list[k]["stg_1"]["real_CTU"] = CU_Y

            # Save entry in final list
            mod_list.append(orig_list[k])

            utils.echo("Complete: {per:.0%}".format(per=k/data_size))

            #
            if (k+1) % accum == 0:
                # Save list to file with the same name
                new_path = os.path.join(new_dir, f[:-4]+str(ctu_count)+ "_" + name_mod)
                dataset_utils.lst2file(mod_list, new_path)

                # Reset list
                mod_list = []

                # Incremente counter
                ctu_count += 1

        if (k - (len(orig_list) - 1)) != 0:  # Verify if there is still entries
            # Save list to file with the same name
            new_path = os.path.join(new_dir, f[:-4] + str(ctu_count) + "_" + name_mod)
            dataset_utils.lst2file(mod_list, new_path)

        # Reset counter
        ctu_count = 0

if __name__ == "__main__":
    main()
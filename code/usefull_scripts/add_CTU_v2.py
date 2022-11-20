import dataset_utils
import MSECNN
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import CustomDataset
import os
import time
import utils
import shutil


"""
This version solves memory problems of the previous version
"""

def main():
    path_dir_l = r"C:\Users\Raul\Dropbox\Dataset\Img\Test_place\delete_later_2\labels_pickle"  # Path with the labels processed
    path_dir_p = r"C:\Users\Raul\Dropbox\Dataset\Img\Test_place\test"  # Path with the pictures

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_real_CTU/"
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
    accum = 1000

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)

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
            cu_pos = (orig_list[k]["CTU"]["cu_pos"]["CU_loc_top"], orig_list[k]["CTU"]["cu_pos"]["CU_loc_left"])
            cu_size = (orig_list[k]["CTU"]["cu_size"]["height"], orig_list[k]["CTU"]["cu_size"]["width"])

            # Get CU
            frame_CU, CU_Y, CU_U, CU_V = CustomDataset.get_cu(img_path, f_size, cu_pos, cu_size, POC)

            # Convert to Pytorch Tensor
            CU_Y = torch.tensor(CU_Y.tolist())

            # Convert to type
            CU_Y = CU_Y.to(dtype=torch.float32)

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Add to entry
            orig_list[k]["real_CTU"] = CU_Y

            # Save entry in final list
            mod_list.append(orig_list[k])

            utils.echo("Complete: {per:.0%}".format(per=k/data_size))

            #
            if (k+1) % accum == 0:
                # Save list to file with the same name
                new_path = os.path.join(new_dir, f[:-4]+str(ctu_count))
                dataset_utils.lst2file(mod_list, new_path)

                # Reset list
                mod_list = []

                # Incremente counter
                ctu_count += 1

        if (k - (len(orig_list) - 1)) != 0:  # Verify if there is still entries
            # Save list to file with the same name
            new_path = os.path.join(new_dir, f[:-4] + str(ctu_count))
            dataset_utils.lst2file(mod_list, new_path)

        # Reset counter
        ctu_count = 0

if __name__ == "__main__":
    main()
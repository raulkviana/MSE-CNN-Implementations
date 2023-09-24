
# Code to process the images/videos into labels
All of the code bellow can be found in this src/msecnn_raulkviana/useful_scripts.

### Encoding
``` python
import dataset_utils

# Quantization parameter
QP = 27

# Temporal Subsample Ratio
ts = 500

# Directory containing the dataset images
d_path = "path_for_images_or_videos"

# Directory containing the .exe file extension to run the encoder.
# It ends with CPIV\VTM-7.0_Data\bin\vs16\msvc-19.24\x86_64\release in Windows
e_path = "path_to_encoder" # Example for windows

dataset_utils.encode_dataset(d_path=d_path, e_path=e_path, ts=ts, QP=QP) # The result is saved in the encoder folder

```

### Generate data structures
``` python
import dataset_utils

# Directory containing the .dat files with CUs informations
d_path = "path_of_output_directory_of_the_previous_step"

dataset_utils.unite_labels_v6_mod(dir_path_l=d_path, n_output_file="modifier_of_the_output_files_name")  # The result is saved in the same folder
```

### Add real CTUs
``` python
import dataset_utils

# Path with the labels processed
path_dir_l = r"path_of_output_directory_of_the_previous_step"  

# Path with the pictures
path_dir_p = r"path_for_images_or_videos"

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
```

### Retrieve essential data
Retrieve essential data for the for stage 6

``` python
import dataset_utils

path_dir_l = "directory_with_the_output_of_the_previous_step"

print("Modifying struct from:", path_dir_l)

dataset_utils.change_struct_no_dupl_stg_6_complexity_v4(path_dir_l)
```

### Balacing data
``` python
import dataset_utils

# Directory containing the .txt files with CUs informations
path_dir_l = "output_of_the_previous_step"  # Path with the labels processed

print("Balance data in:", path_dir_l)

dataset_utils.balance_dataset_down_v3(path_dir_l)  # Downsampling
```
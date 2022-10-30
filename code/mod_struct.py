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

def change_struct_64x64_eval(path_dir_l):

    """
        This version is meant to be used in to process the stage 1 and 2 data
    """
    print("Active function: change_struct_64x64_eval")
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_eval_64x64/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    
    def right_rows(row):
        
        if type(row["stg_2"]) != list:  # Verify that the stage is not empty
            return True
        else:
            return False

    def right_size(row): # Look for CUs with specific size
        
        if (row["stg_2"]["cu_size"]["width"] == 64 and row["stg_3"]["cu_size"]["height"] == 64):
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")


    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        bool_list = list(map(right_size, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        data_size = len(orig_list)

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)
        # Loop entries
        for k in range(data_size):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            if  len(pd_full[(pd_full["POC"] ==  CU_stg2["POC"]) & (pd_full["pic_name"] == CU_stg2["pic_name"]) & \
            (pd_full["cu_size_w"] == CU_stg2["cu_size"]["width"]) & \
            (pd_full["cu_size_h"] == CU_stg2["cu_size"]["height"]) & \
            (pd_full["cu_pos_y"] == CU_stg2["cu_pos"]["CU_loc_top"]) &\
            (pd_full["cu_pos_x"] == CU_stg2["cu_pos"]["CU_loc_left"])]) == 0 : 
                cu_pos = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg2['RD0']
                RD1 = CU_stg2['RD1']
                RD2 = CU_stg2['RD2']
                RD3 = CU_stg2['RD3']
                RD4 = CU_stg2['RD4']
                RD5 = CU_stg2['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split = CU_stg2["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos"] = cu_pos
                orig_list[k]["cu_size"] = cu_size
                orig_list[k]["split"] = split
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU
                orig_list[k]["orig_pos_x"] = CU_stg2["cu_pos"]["CU_loc_left"]
                orig_list[k]["orig_pos_y"] = CU_stg2["cu_pos"]["CU_loc_top"]
                orig_list[k]["orig_size_h"] = CU_stg2["cu_size"]["height"]
                orig_list[k]["orig_size_w"] = CU_stg2["cu_size"]["width"]
    
                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                # Save entry in final list
                mod_list.append(orig_list[k])

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg2["POC"]], "pic_name": CU_stg2["pic_name"], "cu_pos_x": CU_stg2["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg2["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg2["cu_size"]["height"],\
                         "cu_size_w": CU_stg2["cu_size"]["width"], "split": CU_stg2["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

            # Save list to file with the same name
            new_path = os.path.join(new_dir, f[:-4])
            dataset_utils.lst2file(mod_list, new_path)


def change_struct_32x32_eval(path_dir_l):

    """
        This version is meant to be used in to process the stage 3 data
    """
    print("Active function: change_struct_32x32_eval")
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_eval_32x32/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)
    
    def right_rows(row):
        
        if type(row["stg_3"]) != list:  # Verify that the stage is not empty
            return True
        else:
            return False

    def right_size(row): # Look for CUs with specific size
        
        if (row["stg_3"]["cu_size"]["width"] == 32 and row["stg_3"]["cu_size"]["height"] == 32):
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        bool_list = list(map(right_size, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        data_size = len(orig_list)

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(data_size):

            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]

            if  len(pd_full[(pd_full["POC"] ==  CU_stg3["POC"]) & (pd_full["pic_name"] == CU_stg3["pic_name"]) & \
            (pd_full["cu_size_w"] == CU_stg3["cu_size"]["width"]) & \
            (pd_full["cu_size_h"] == CU_stg3["cu_size"]["height"]) & \
            (pd_full["cu_pos_y"] == CU_stg3["cu_pos"]["CU_loc_top"]) &\
            (pd_full["cu_pos_x"] == CU_stg3["cu_pos"]["CU_loc_left"])]) == 0 : 

                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"], CU_stg3["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg3['RD0']
                RD1 = CU_stg3['RD1']
                RD2 = CU_stg3['RD2']
                RD3 = CU_stg3['RD3']
                RD4 = CU_stg3['RD4']
                RD5 = CU_stg3['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split"] = split_stg3
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU
                orig_list[k]["orig_pos_x"] = CU_stg3["cu_pos"]["CU_loc_left"]
                orig_list[k]["orig_pos_y"] = CU_stg3["cu_pos"]["CU_loc_top"]
                orig_list[k]["orig_size_h"] = CU_stg3["cu_size"]["height"]
                orig_list[k]["orig_size_w"] = CU_stg3["cu_size"]["width"]
    
                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                # Save entry in final list
                mod_list.append(orig_list[k])

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg3["POC"]], "pic_name": CU_stg3["pic_name"], "cu_pos_x": CU_stg3["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg3["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg3["cu_size"]["height"],\
                         "cu_size_w": CU_stg3["cu_size"]["width"], "split": CU_stg3["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4])
        dataset_utils.lst2file(mod_list, new_path)

def change_struct_64x64(path_dir_l):

    """
        This version is meant to be used in to process the stage 1 and 2 data
    """
    print("Active function: change_struct_64x64")

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

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
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU = orig_list[k]["stg_2"]
            cu_pos = torch.reshape(torch.tensor([CU["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                   CU["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_size = torch.reshape(torch.tensor([CU["cu_size"]["height"], CU["cu_size"]["width"]]), (1,-1))
            RD0 = CU['RD0']
            RD1 = CU['RD1']
            RD2 = CU['RD2']
            RD3 = CU['RD3']
            RD4 = CU['RD4']
            RD5 = CU['RD5']
            RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
            split = CU["split"]
            real_CTU = CTU["real_CTU"]

            # Add new entries
            orig_list[k]["RD"] = RD_tens
            orig_list[k]["cu_pos"] = cu_pos
            orig_list[k]["cu_size"] = cu_size
            orig_list[k]["split"] = split
            orig_list[k]["real_CTU"] = real_CTU
            orig_list[k]["CTU"] = CTU

            # Delete unnecessary entries
            for k2 in range(6):
                del orig_list[k]["stg_"+str(k2+1)]

            del orig_list[k]["POC"]
            del orig_list[k]["pic_name"]

            # Save entry in final list
            mod_list.append(orig_list[k])

            utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4])
        dataset_utils.lst2file(mod_list, new_path)


def change_struct_64x64_no_dupl_v3(path_dir_l):
    """
        This version is like the change_struct_64x64_no_dupl_v2, with threads
    """
    print("Active function: change_struct_64x64_no_dupl_v3")

    # Save time
    t0 = time.time()
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_64x64_v3/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)
    #    pass

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_2"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    # Process files
    threads = []
    for f in files_l:
        x = threading.Thread(target=mod_64x64_threads, args=(f, path_dir_l, right_rows, columns, new_dir))
        threads.append(x)
        x.start()

    for thread in threads:
        thread.join()

    print("Time elapsed: ", time.time() - t0)


def mod_64x64_threads(f, path_dir_l, right_rows, columns, new_dir):
    
    # Make labels path
    lbls_path = os.path.join(path_dir_l, f)

    # List to save entries
    mod_list = []

    # Read file
    orig_list = dataset_utils.file2lst(lbls_path[:-4])
    data_size = len(orig_list)
    #orig_frame = pd.DataFrame(orig_list)
    bool_list = list(map(right_rows, orig_list))
    idx_list = list(np.where(bool_list)[0])
    orig_list = list(map(orig_list.__getitem__, idx_list))

    #(type(orig_frame["stg_3"]) != list) & (type(orig_frame["stg_4"]) == list)

    # Dataframe initialization
    pd_full = pd.DataFrame(columns=columns)

    # Verbose
    print("Processing:", lbls_path)

    # Loop entries
    for k in range(len(orig_list)):
        
        # New entries
        CTU = orig_list[k]["stg_1"]
        CU_stg2 = orig_list[k]["stg_2"]
        
        # Verify size of the CU and if the variable structure is of what type
        if type(CU_stg2) == list or CU_stg2["cu_size"]["width"] != 64 or CU_stg2["cu_size"]["height"] != 64:
            continue

        # Verify if cu wasn't added already
        if  len(pd_full[(pd_full["POC"] ==  CU_stg2["POC"]) & (pd_full["pic_name"] == CU_stg2["pic_name"]) & \
            (pd_full["cu_size_w"] == CU_stg2["cu_size"]["width"]) & \
            (pd_full["cu_size_h"] == CU_stg2["cu_size"]["height"]) & \
            (pd_full["cu_pos_y"] == CU_stg2["cu_pos"]["CU_loc_top"]) &\
            (pd_full["cu_pos_x"] == CU_stg2["cu_pos"]["CU_loc_left"])]) == 0 : 
                
            cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
            RD0 = CU_stg2['RD0']
            RD1 = CU_stg2['RD1']
            RD2 = CU_stg2['RD2']
            RD3 = CU_stg2['RD3']
            RD4 = CU_stg2['RD4']
            RD5 = CU_stg2['RD5']
            RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
            split_stg2 = CU_stg2["split"]
            real_CTU = CTU["real_CTU"]

            # Add new entries
            orig_list[k]["RD"] = RD_tens
            orig_list[k]["cu_pos"] = cu_pos_stg2
            orig_list[k]["cu_size"] = cu_size_stg2
            orig_list[k]["split"] = split_stg2
            orig_list[k]["real_CTU"] = real_CTU
            orig_list[k]["CTU"] = CTU

            # Update dataframe
            pd_row = pd.DataFrame({"POC": [CU_stg2["POC"]], "pic_name": CU_stg2["pic_name"], "cu_pos_x": CU_stg2["cu_pos"]["CU_loc_left"],\
                        "cu_pos_y": CU_stg2["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg2["cu_size"]["height"],\
                        "cu_size_w": CU_stg2["cu_size"]["width"], "split": CU_stg2["split"]})
            pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

            # Delete unnecessary entries
            for k2 in range(6):
                del orig_list[k]["stg_"+str(k2+1)]

            del orig_list[k]["POC"]
            del orig_list[k]["pic_name"]

            # Save entry in final list
            mod_list.append(orig_list[k])

            utils.echo("Complete: {per:.0%}".format(per=k/data_size))

    # Save list to file with the same name
    new_path = os.path.join(new_dir, f[:-4]+"_mod_with_struct")
    dataset_utils.lst2file(mod_list, new_path)


def change_struct_64x64_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2.
    """
    print("Active function: change_struct_64x64_no_dupl_v2")

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_64x64_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)
    #    pass

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_2"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))

        #(type(orig_frame["stg_3"]) != list) & (type(orig_frame["stg_4"]) == list)

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            
            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg2) == list or CU_stg2["cu_size"]["width"] != 64 or CU_stg2["cu_size"]["height"] != 64:
                continue

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg2["POC"]) & (pd_full["pic_name"] == CU_stg2["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg2["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg2["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg2["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg2["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg2['RD0']
                RD1 = CU_stg2['RD1']
                RD2 = CU_stg2['RD2']
                RD3 = CU_stg2['RD3']
                RD4 = CU_stg2['RD4']
                RD5 = CU_stg2['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos"] = cu_pos_stg2
                orig_list[k]["cu_size"] = cu_size_stg2
                orig_list[k]["split"] = split_stg2
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg2["POC"]], "pic_name": CU_stg2["pic_name"], "cu_pos_x": CU_stg2["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg2["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg2["cu_size"]["height"],\
                         "cu_size_w": CU_stg2["cu_size"]["width"], "split": CU_stg2["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"_mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)


def change_struct_32x32(path_dir_l):
    """
        This version is meant to be used in to process the stage 3 data
    """
    print("Active function: change_struct_32x32")

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_32x32/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

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
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            
            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg3) == list:
                continue

            elif CU_stg3["cu_size"]["width"] != 32 or CU_stg3["cu_size"]["height"] != 32:
                continue
                
            cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                   CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                   CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
            cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
            RD0 = CU_stg3['RD0']
            RD1 = CU_stg3['RD1']
            RD2 = CU_stg3['RD2']
            RD3 = CU_stg3['RD3']
            RD4 = CU_stg3['RD4']
            RD5 = CU_stg3['RD5']
            RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
            split_stg2 = CU_stg2["split"]
            split_stg3 = CU_stg3["split"]
            real_CTU = CTU["real_CTU"]

            # Add new entries
            orig_list[k]["RD"] = RD_tens
            orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
            orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
            orig_list[k]["cu_size_stg2"] = cu_size_stg2
            orig_list[k]["cu_size_stg3"] = cu_size_stg3
            orig_list[k]["split_stg2"] = split_stg2
            orig_list[k]["split"] = split_stg3
            orig_list[k]["real_CTU"] = real_CTU
            orig_list[k]["CTU"] = CTU

            # Delete unnecessary entries
            for k2 in range(6):
                del orig_list[k]["stg_"+str(k2+1)]

            del orig_list[k]["POC"]
            del orig_list[k]["pic_name"]

            # Save entry in final list
            mod_list.append(orig_list[k])

            utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4])
        dataset_utils.lst2file(mod_list, new_path)

def change_struct_32x32_no_dupl(path_dir_l):
    """
        This version is like the change_struct_32x32, but it removes possible duplicated rows.
    """
    print("Active function: change_struct_32x32_no_dupl")

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_32x32/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)
    #    pass

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            
            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg3) == list:
                continue

            elif CU_stg3["cu_size"]["width"] != 32 or CU_stg3["cu_size"]["height"] != 32:
                continue

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg3["POC"]) & (pd_full["pic_name"] == CU_stg3["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg3["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg3["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg3["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg3["cu_pos"]["CU_loc_left"]) &\
                (pd_full["split"] == CU_stg3["split"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg3['RD0']
                RD1 = CU_stg3['RD1']
                RD2 = CU_stg3['RD2']
                RD3 = CU_stg3['RD3']
                RD4 = CU_stg3['RD4']
                RD5 = CU_stg3['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split"] = split_stg3
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg3["POC"]], "pic_name": CU_stg3["pic_name"], "cu_pos_x": CU_stg3["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg3["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg3["cu_size"]["height"],\
                         "cu_size_w": CU_stg3["cu_size"]["width"], "split": CU_stg3["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)


def change_struct_32x32_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is smarter.
    """
    print("Active function: change_struct_32x32_no_dupl_v2")

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_32x32_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)
    #    pass

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_3"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))

        #(type(orig_frame["stg_3"]) != list) & (type(orig_frame["stg_4"]) == list)

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            
            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg3) == list or CU_stg3["cu_size"]["width"] != 32 or CU_stg3["cu_size"]["height"] != 32:
                continue

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg3["POC"]) & (pd_full["pic_name"] == CU_stg3["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg3["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg3["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg3["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg3["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg3['RD0']
                RD1 = CU_stg3['RD1']
                RD2 = CU_stg3['RD2']
                RD3 = CU_stg3['RD3']
                RD4 = CU_stg3['RD4']
                RD5 = CU_stg3['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split"] = split_stg3
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg3["POC"]], "pic_name": CU_stg3["pic_name"], "cu_pos_x": CU_stg3["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg3["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg3["cu_size"]["height"],\
                         "cu_size_w": CU_stg3["cu_size"]["width"], "split": CU_stg3["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)


def change_struct_32x32_no_dupl_v3(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but uses threads.
    """
    # Initial steps
    print("Active function: change_struct_32x32_no_dupl_v3")
    t0 = time.time()

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_32x32_v3/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_3"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    # Process files
    threads = []
    for f in files_l:
        x = threading.Thread(target=mod_32x32_threads, args=(f, path_dir_l, right_rows, columns, new_dir))
        threads.append(x)
        x.start()

    for thread in threads:
        thread.join()
   
    print("Time elapsed: ", time.time() - t0)


def mod_32x32_threads(f, path_dir_l, right_rows, columns, new_dir):
    
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))

        #(type(orig_frame["stg_3"]) != list) & (type(orig_frame["stg_4"]) == list)

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            
            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg3) == list or CU_stg3["cu_size"]["width"] != 32 or CU_stg3["cu_size"]["height"] != 32:
                continue

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg3["POC"]) & (pd_full["pic_name"] == CU_stg3["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg3["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg3["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg3["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg3["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg3['RD0']
                RD1 = CU_stg3['RD1']
                RD2 = CU_stg3['RD2']
                RD3 = CU_stg3['RD3']
                RD4 = CU_stg3['RD4']
                RD5 = CU_stg3['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split"] = split_stg3
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg3["POC"]], "pic_name": CU_stg3["pic_name"], "cu_pos_x": CU_stg3["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg3["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg3["cu_size"]["height"],\
                         "cu_size_w": CU_stg3["cu_size"]["width"], "split": CU_stg3["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)


def change_struct_32x32_no_dupl_v2_test(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but is for verifying if everything is right
    """
    print("Active function: change_struct_32x32_no_dupl_v2_test")

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_32x32_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)
    #    pass

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_3"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg3 = orig_list[k]["stg_3"]
            
            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg3) == list or CU_stg3["cu_size"]["width"] != 32 or CU_stg3["cu_size"]["height"] != 32:
                continue

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg3["POC"]) & (pd_full["pic_name"] == CU_stg3["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg3["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg3["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg3["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg3["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg3_y = CU_stg3["cu_pos"]["CU_loc_top"]
                cu_pos_stg3_x = CU_stg3["cu_pos"]["CU_loc_left"]
                cu_size_stg3_h = CU_stg3["cu_size"]["height"] 
                cu_size_stg3_w = CU_stg3["cu_size"]["width"] 
                split_stg3 = CU_stg3["split"]
                POC_stg3 = CTU["POC"]
                pic_name_stg3 = CTU["pic_name"]

                # Add new entries
                orig_list[k]["y"] = cu_pos_stg3_y
                orig_list[k]["x"] = cu_pos_stg3_x
                orig_list[k]["h"] = cu_size_stg3_h
                orig_list[k]["w"] = cu_size_stg3_w
                orig_list[k]["POC"] = POC_stg3
                orig_list[k]["pic_name"] = pic_name_stg3
                orig_list[k]["split"] = split_stg3

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg3["POC"]], "pic_name": CU_stg3["pic_name"], "cu_pos_x": CU_stg3["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg3["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg3["cu_size"]["height"],\
                         "cu_size_w": CU_stg3["cu_size"]["width"], "split": CU_stg3["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

def change_struct_16x16_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to 16x16 CUs.
    """
    # Initial steps
    print("Active function: change_struct_16x16_no_dupl_v2")
    t0 = time.time()
    
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_16x16_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_4"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]
            
            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg4) == list or CU_stg4["cu_size"]["width"] != 16 or CU_stg4["cu_size"]["height"] != 16:
                continue

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg4["POC"]) & (pd_full["pic_name"] == CU_stg4["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg4["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg4["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg4["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg4["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                    CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg4['RD0']
                RD1 = CU_stg4['RD1']
                RD2 = CU_stg4['RD2']
                RD3 = CU_stg4['RD3']
                RD4 = CU_stg4['RD4']
                RD5 = CU_stg4['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                split_stg4 = CU_stg4["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["cu_size_stg4"] = cu_size_stg4
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split_stg3"] = split_stg3
                orig_list[k]["split"] = split_stg4
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg4["POC"]], "pic_name": CU_stg4["pic_name"], "cu_pos_x": CU_stg4["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg4["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg4["cu_size"]["height"],\
                         "cu_size_w": CU_stg4["cu_size"]["width"], "split": CU_stg4["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_8x8_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to 16x16 CUs.
    """
    # Initial steps
    print("Active function: change_struct_8x8_no_dupl_v2")
    t0 = time.time()
    
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_8x8_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_5"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))

        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]
            CU_stg5 = orig_list[k]["stg_5"]
            
            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg5) == list or CU_stg5["cu_size"]["width"] != 8 or CU_stg5["cu_size"]["height"] != 8:
                continue

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg5["POC"]) & (pd_full["pic_name"] == CU_stg5["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg5["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg5["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg5["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg5["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                    CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_pos"]["CU_loc_top"] - CU_stg4["cu_pos"]["CU_loc_top"],
                                    CU_stg5["cu_pos"]["CU_loc_left"] - CU_stg4["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
                cu_size_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_size"]["height"],CU_stg5["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg5['RD0']
                RD1 = CU_stg5['RD1']
                RD2 = CU_stg5['RD2']
                RD3 = CU_stg5['RD3']
                RD4 = CU_stg5['RD4']
                RD5 = CU_stg5['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                split_stg4 = CU_stg4["split"]
                split_stg5 = CU_stg5["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
                orig_list[k]["cu_pos_stg5"] = cu_pos_stg5
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["cu_size_stg4"] = cu_size_stg4
                orig_list[k]["cu_size_stg5"] = cu_size_stg5
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split_stg3"] = split_stg3
                orig_list[k]["split_stg4"] = split_stg4
                orig_list[k]["split"] = split_stg5
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg5["POC"]], "pic_name": CU_stg5["pic_name"], "cu_pos_x": CU_stg5["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg5["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg5["cu_size"]["height"],\
                         "cu_size_w": CU_stg5["cu_size"]["width"], "split": CU_stg5["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg6_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 6
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg6_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg6_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_3']) != list]  # Remove empty stage
        orig_list = [d for d in orig_list if type(d['stg_4']) != list]  # Remove empty stage
        orig_list = [d for d in orig_list if type(d['stg_5']) != list]  # Remove empty stage
        orig_list = [d for d in orig_list if type(d['stg_6']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_6"]['RD0'], l["stg_6"]['RD1'], l["stg_6"]['RD2'], l["stg_6"]['RD3'], l["stg_6"]['RD4'], l["stg_6"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_3"]["cu_pos"]["CU_loc_top"] - l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"] - l["stg_2"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_4"]["cu_pos"]["CU_loc_top"] - l["stg_3"]["cu_pos"]["CU_loc_top"], l["stg_4"]["cu_pos"]["CU_loc_left"] - l["stg_3"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_5"]["cu_pos"]["CU_loc_top"] - l["stg_4"]["cu_pos"]["CU_loc_top"], l["stg_5"]["cu_pos"]["CU_loc_left"] - l["stg_4"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_6"]["cu_pos"]["CU_loc_top"] - l["stg_5"]["cu_pos"]["CU_loc_top"], l["stg_6"]["cu_pos"]["CU_loc_left"] - l["stg_5"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                               [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]], \
                               [l["stg_4"]["cu_size"]["height"], l["stg_4"]["cu_size"]["width"]], \
                               [l["stg_5"]["cu_size"]["height"], l["stg_5"]["cu_size"]["width"]], \
                               [l["stg_6"]["cu_size"]["height"], l["stg_6"]["cu_size"]["width"]], \
                                  l["stg_2"]["split"], l["stg_3"]["split"], l["stg_4"]["split"], l["stg_5"]["split"], l["stg_6"]["split"], \
                                  l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)

def change_struct_no_dupl_stg5_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 5
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg5_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg5_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_3']) != list]  # Remove empty stage
        orig_list = [d for d in orig_list if type(d['stg_4']) != list]  # Remove empty stage
        orig_list = [d for d in orig_list if type(d['stg_5']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_5"]['RD0'], l["stg_5"]['RD1'], l["stg_5"]['RD2'], l["stg_5"]['RD3'], l["stg_5"]['RD4'], l["stg_5"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_3"]["cu_pos"]["CU_loc_top"] - l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"] - l["stg_2"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_4"]["cu_pos"]["CU_loc_top"] - l["stg_3"]["cu_pos"]["CU_loc_top"], l["stg_4"]["cu_pos"]["CU_loc_left"] - l["stg_3"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_5"]["cu_pos"]["CU_loc_top"] - l["stg_4"]["cu_pos"]["CU_loc_top"], l["stg_5"]["cu_pos"]["CU_loc_left"] - l["stg_4"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                               [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]], \
                               [l["stg_4"]["cu_size"]["height"], l["stg_4"]["cu_size"]["width"]], \
                               [l["stg_5"]["cu_size"]["height"], l["stg_5"]["cu_size"]["width"]], \
                                  l["stg_2"]["split"], l["stg_3"]["split"], l["stg_4"]["split"], l["stg_5"]["split"], \
                                  l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)

def change_struct_no_dupl_stg2_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 2
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg2_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg2_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_2']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_2"]['RD0'], l["stg_2"]['RD1'], l["stg_2"]['RD2'], l["stg_2"]['RD3'], l["stg_2"]['RD4'], l["stg_2"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                                  l["stg_2"]["split"], \
                                  l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)

def change_struct_no_dupl_stg4_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 4
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg4_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg4_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_3']) != list]  # Remove empty stage
        orig_list = [d for d in orig_list if type(d['stg_4']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_4"]['RD0'], l["stg_4"]['RD1'], l["stg_4"]['RD2'], l["stg_4"]['RD3'], l["stg_4"]['RD4'], l["stg_4"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_3"]["cu_pos"]["CU_loc_top"] - l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"] - l["stg_2"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_4"]["cu_pos"]["CU_loc_top"] - l["stg_3"]["cu_pos"]["CU_loc_top"], l["stg_4"]["cu_pos"]["CU_loc_left"] - l["stg_3"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                               [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]], \
                               [l["stg_4"]["cu_size"]["height"], l["stg_4"]["cu_size"]["width"]], \
                                  l["stg_2"]["split"], l["stg_3"]["split"], l["stg_4"]["split"], \
                                  l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg3_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 3
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg3_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg3_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_3']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_3"]['RD0'], l["stg_3"]['RD1'], l["stg_3"]['RD2'], l["stg_3"]['RD3'], l["stg_3"]['RD4'], l["stg_3"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_3"]["cu_pos"]["CU_loc_top"] - l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"] - l["stg_2"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                               [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]], \
                               l["stg_2"]["split"], l["stg_3"]["split"], \
                               l["stg_1"]["real_CTU"].tolist(), l["stg_1"]["color_ch"], l["POC"], l["pic_name"]  ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_32x16_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to 32x16 CUs.
    """
    # Initial steps
    print("Active function: change_struct_32x16_no_dupl_v2")
    t0 = time.time()
    
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_32x16_v2_2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_4"]) != list:  # Verify that the stage is not empty
            return True
        else:
            return False

    def right_size(row): # Look for CUs with specific size
        
        if (row["stg_4"]["cu_size"]["width"] == 16 and row["stg_4"]["cu_size"]["height"] == 32) or (row["stg_4"]["cu_size"]["width"] == 32 and row["stg_4"]["cu_size"]["height"] == 16):  # Verify that the stage is not empty
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        bool_list = list(map(right_size, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg4["POC"]) & (pd_full["pic_name"] == CU_stg4["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg4["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg4["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg4["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg4["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                    CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg4['RD0']
                RD1 = CU_stg4['RD1']
                RD2 = CU_stg4['RD2']
                RD3 = CU_stg4['RD3']
                RD4 = CU_stg4['RD4']
                RD5 = CU_stg4['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                split_stg4 = CU_stg4["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["cu_size_stg4"] = cu_size_stg4
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split_stg3"] = split_stg3
                orig_list[k]["split"] = split_stg4
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg4["POC"]], "pic_name": CU_stg4["pic_name"], "cu_pos_x": CU_stg4["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg4["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg4["cu_size"]["height"],\
                         "cu_size_w": CU_stg4["cu_size"]["width"], "split": CU_stg4["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_32x8_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to 32x8 CUs.
    """
    # Initial steps
    print("Active function: change_struct_32x8_no_dupl_v2")
    t0 = time.time()
    
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_32x8_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_5"]) != list:  # Verify that the stage is not empty
            return True
        else:
            return False

    def right_size(row): # Look for CUs with specific size
        
        if (row["stg_5"]["cu_size"]["width"] == 8 and row["stg_5"]["cu_size"]["height"] == 32) or (row["stg_5"]["cu_size"]["width"] == 32 and row["stg_5"]["cu_size"]["height"] == 8):  # Verify that the stage is not empty
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        bool_list = list(map(right_size, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]
            CU_stg5 = orig_list[k]["stg_5"]

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg5["POC"]) & (pd_full["pic_name"] == CU_stg5["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg5["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg5["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg5["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg5["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                    CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_pos"]["CU_loc_top"] - CU_stg4["cu_pos"]["CU_loc_top"],
                                    CU_stg5["cu_pos"]["CU_loc_left"] - CU_stg4["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
                cu_size_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_size"]["height"],CU_stg5["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg5['RD0']
                RD1 = CU_stg5['RD1']
                RD2 = CU_stg5['RD2']
                RD3 = CU_stg5['RD3']
                RD4 = CU_stg5['RD4']
                RD5 = CU_stg5['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                split_stg4 = CU_stg4["split"]
                split_stg5 = CU_stg5["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
                orig_list[k]["cu_pos_stg5"] = cu_pos_stg5
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["cu_size_stg4"] = cu_size_stg4
                orig_list[k]["cu_size_stg5"] = cu_size_stg5
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split_stg3"] = split_stg3
                orig_list[k]["split_stg4"] = split_stg4
                orig_list[k]["split"] = split_stg5
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg5["POC"]], "pic_name": CU_stg5["pic_name"], "cu_pos_x": CU_stg5["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg5["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg5["cu_size"]["height"],\
                         "cu_size_w": CU_stg5["cu_size"]["width"], "split": CU_stg5["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)

def change_struct_16x8_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to 16x8 CUs.
    """
    # Initial steps
    print("Active function: change_struct_16x8_no_dupl_v2")
    t0 = time.time()
    
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_16x8_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_5"]) != list:  # Verify that the stage is not empty
            return True
        else:
            return False

    def right_size(row): # Look for CUs with specific size
        
        if (row["stg_5"]["cu_size"]["width"] == 8 and row["stg_5"]["cu_size"]["height"] == 16) or (row["stg_5"]["cu_size"]["width"] == 16 and row["stg_5"]["cu_size"]["height"] == 8):  # Verify that the stage is not empty
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        bool_list = list(map(right_size, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]
            CU_stg5 = orig_list[k]["stg_5"]

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg5["POC"]) & (pd_full["pic_name"] == CU_stg5["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg5["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg5["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg5["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg5["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                    CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_pos"]["CU_loc_top"] - CU_stg4["cu_pos"]["CU_loc_top"],
                                    CU_stg5["cu_pos"]["CU_loc_left"] - CU_stg4["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
                cu_size_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_size"]["height"],CU_stg5["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg5['RD0']
                RD1 = CU_stg5['RD1']
                RD2 = CU_stg5['RD2']
                RD3 = CU_stg5['RD3']
                RD4 = CU_stg5['RD4']
                RD5 = CU_stg5['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                split_stg4 = CU_stg4["split"]
                split_stg5 = CU_stg5["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
                orig_list[k]["cu_pos_stg5"] = cu_pos_stg5
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["cu_size_stg4"] = cu_size_stg4
                orig_list[k]["cu_size_stg5"] = cu_size_stg5
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split_stg3"] = split_stg3
                orig_list[k]["split_stg4"] = split_stg4
                orig_list[k]["split"] = split_stg5
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg5["POC"]], "pic_name": CU_stg5["pic_name"], "cu_pos_x": CU_stg5["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg5["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg5["cu_size"]["height"],\
                         "cu_size_w": CU_stg5["cu_size"]["width"], "split": CU_stg5["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_8x4_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to 8x4 CUs.
    """
    # Initial steps
    print("Active function: change_struct_8x4_no_dupl_v2")
    t0 = time.time()
    
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_8x4_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_6"]) != list:  # Verify that the stage is not empty
            return True
        else:
            return False

    def right_size(row): # Look for CUs with specific size
        
        if (row["stg_6"]["cu_size"]["width"] == 4 and row["stg_6"]["cu_size"]["height"] == 8) or (row["stg_6"]["cu_size"]["width"] == 8 and row["stg_6"]["cu_size"]["height"] == 4):  # Verify that the stage is not empty
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        bool_list = list(map(right_size, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]
            CU_stg5 = orig_list[k]["stg_5"]
            CU_stg6 = orig_list[k]["stg_6"]

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg6["POC"]) & (pd_full["pic_name"] == CU_stg6["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg6["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg6["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg6["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg6["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                    CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_pos"]["CU_loc_top"] - CU_stg4["cu_pos"]["CU_loc_top"],
                                    CU_stg5["cu_pos"]["CU_loc_left"] - CU_stg4["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg6 = torch.reshape(torch.tensor([CU_stg6["cu_pos"]["CU_loc_top"] - CU_stg5["cu_pos"]["CU_loc_top"],
                                    CU_stg6["cu_pos"]["CU_loc_left"] - CU_stg5["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
                cu_size_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_size"]["height"],CU_stg5["cu_size"]["width"]]), (1,-1))
                cu_size_stg6 = torch.reshape(torch.tensor([CU_stg6["cu_size"]["height"],CU_stg6["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg6['RD0']
                RD1 = CU_stg6['RD1']
                RD2 = CU_stg6['RD2']
                RD3 = CU_stg6['RD3']
                RD4 = CU_stg6['RD4']
                RD5 = CU_stg6['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                split_stg4 = CU_stg4["split"]
                split_stg5 = CU_stg5["split"]
                split_stg6 = CU_stg6["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
                orig_list[k]["cu_pos_stg5"] = cu_pos_stg5
                orig_list[k]["cu_pos_stg6"] = cu_pos_stg6
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["cu_size_stg4"] = cu_size_stg4
                orig_list[k]["cu_size_stg5"] = cu_size_stg5
                orig_list[k]["cu_size_stg6"] = cu_size_stg6
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split_stg3"] = split_stg3
                orig_list[k]["split_stg4"] = split_stg4
                orig_list[k]["split_stg5"] = split_stg5
                orig_list[k]["split"] = split_stg6
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg6["POC"]], "pic_name": CU_stg6["pic_name"], "cu_pos_x": CU_stg6["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg6["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg6["cu_size"]["height"],\
                         "cu_size_w": CU_stg6["cu_size"]["width"], "split": CU_stg6["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_32x4_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to 32x4 CUs.
    """
    # Initial steps
    print("Active function: change_struct_32x4_no_dupl_v2")
    t0 = time.time()
    
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_32x4_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_6"]) != list:  # Verify that the stage is not empty
            return True
        else:
            return False

    def right_size(row): # Look for CUs with specific size
        
        if (row["stg_6"]["cu_size"]["width"] == 4 and row["stg_6"]["cu_size"]["height"] == 32) or (row["stg_6"]["cu_size"]["width"] == 32 and row["stg_6"]["cu_size"]["height"] == 4):  # Verify that the stage is not empty
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        bool_list = list(map(right_size, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]
            CU_stg5 = orig_list[k]["stg_5"]
            CU_stg6 = orig_list[k]["stg_6"]

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg6["POC"]) & (pd_full["pic_name"] == CU_stg6["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg6["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg6["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg6["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg6["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                    CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_pos"]["CU_loc_top"] - CU_stg4["cu_pos"]["CU_loc_top"],
                                    CU_stg5["cu_pos"]["CU_loc_left"] - CU_stg4["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg6 = torch.reshape(torch.tensor([CU_stg6["cu_pos"]["CU_loc_top"] - CU_stg5["cu_pos"]["CU_loc_top"],
                                    CU_stg6["cu_pos"]["CU_loc_left"] - CU_stg5["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
                cu_size_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_size"]["height"],CU_stg5["cu_size"]["width"]]), (1,-1))
                cu_size_stg6 = torch.reshape(torch.tensor([CU_stg6["cu_size"]["height"],CU_stg6["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg6['RD0']
                RD1 = CU_stg6['RD1']
                RD2 = CU_stg6['RD2']
                RD3 = CU_stg6['RD3']
                RD4 = CU_stg6['RD4']
                RD5 = CU_stg6['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                split_stg4 = CU_stg4["split"]
                split_stg5 = CU_stg5["split"]
                split_stg6 = CU_stg6["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
                orig_list[k]["cu_pos_stg5"] = cu_pos_stg5
                orig_list[k]["cu_pos_stg6"] = cu_pos_stg6
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["cu_size_stg4"] = cu_size_stg4
                orig_list[k]["cu_size_stg5"] = cu_size_stg5
                orig_list[k]["cu_size_stg6"] = cu_size_stg6
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split_stg3"] = split_stg3
                orig_list[k]["split_stg4"] = split_stg4
                orig_list[k]["split_stg5"] = split_stg5
                orig_list[k]["split"] = split_stg6
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg6["POC"]], "pic_name": CU_stg6["pic_name"], "cu_pos_x": CU_stg6["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg6["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg6["cu_size"]["height"],\
                         "cu_size_w": CU_stg6["cu_size"]["width"], "split": CU_stg6["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)

def change_struct_16x4_no_dupl_v2(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to 8x4 CUs.
    """
    # Initial steps
    print("Active function: change_struct_16x4_no_dupl_v2")
    t0 = time.time()
    
    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_16x4_v2/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_6"]) != list:  # Verify that the stage is not empty
            return True
        else:
            return False

    def right_size(row): # Look for CUs with specific size
        
        if (row["stg_6"]["cu_size"]["width"] == 4 and row["stg_6"]["cu_size"]["height"] == 16) or (row["stg_6"]["cu_size"]["width"] == 16 and row["stg_6"]["cu_size"]["height"] == 4):  # Verify that the stage is not empty
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        data_size = len(orig_list)
        #orig_frame = pd.DataFrame(orig_list)
        bool_list = list(map(right_rows, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        bool_list = list(map(right_size, orig_list))
        idx_list = list(np.where(bool_list)[0])
        orig_list = list(map(orig_list.__getitem__, idx_list))
        # Dataframe initialization
        pd_full = pd.DataFrame(columns=columns)

        # Verbose
        print("Processing:", lbls_path)

        # Loop entries
        for k in range(len(orig_list)):
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]
            CU_stg5 = orig_list[k]["stg_5"]
            CU_stg6 = orig_list[k]["stg_6"]

            # Verify if cu wasn't added already
            if  len(pd_full[(pd_full["POC"] ==  CU_stg6["POC"]) & (pd_full["pic_name"] == CU_stg6["pic_name"]) & \
                (pd_full["cu_size_w"] == CU_stg6["cu_size"]["width"]) & \
                (pd_full["cu_size_h"] == CU_stg6["cu_size"]["height"]) & \
                (pd_full["cu_pos_y"] == CU_stg6["cu_pos"]["CU_loc_top"]) &\
                (pd_full["cu_pos_x"] == CU_stg6["cu_pos"]["CU_loc_left"])]) == 0 : 
                    
                cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                    CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                    CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                    CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_pos"]["CU_loc_top"] - CU_stg4["cu_pos"]["CU_loc_top"],
                                    CU_stg5["cu_pos"]["CU_loc_left"] - CU_stg4["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_pos_stg6 = torch.reshape(torch.tensor([CU_stg6["cu_pos"]["CU_loc_top"] - CU_stg5["cu_pos"]["CU_loc_top"],
                                    CU_stg6["cu_pos"]["CU_loc_left"] - CU_stg5["cu_pos"]["CU_loc_left"]]), (1,-1))
                cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
                cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
                cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
                cu_size_stg5 = torch.reshape(torch.tensor([CU_stg5["cu_size"]["height"],CU_stg5["cu_size"]["width"]]), (1,-1))
                cu_size_stg6 = torch.reshape(torch.tensor([CU_stg6["cu_size"]["height"],CU_stg6["cu_size"]["width"]]), (1,-1))
                RD0 = CU_stg6['RD0']
                RD1 = CU_stg6['RD1']
                RD2 = CU_stg6['RD2']
                RD3 = CU_stg6['RD3']
                RD4 = CU_stg6['RD4']
                RD5 = CU_stg6['RD5']
                RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
                split_stg2 = CU_stg2["split"]
                split_stg3 = CU_stg3["split"]
                split_stg4 = CU_stg4["split"]
                split_stg5 = CU_stg5["split"]
                split_stg6 = CU_stg6["split"]
                real_CTU = CTU["real_CTU"]

                # Add new entries
                orig_list[k]["RD"] = RD_tens
                orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
                orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
                orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
                orig_list[k]["cu_pos_stg5"] = cu_pos_stg5
                orig_list[k]["cu_pos_stg6"] = cu_pos_stg6
                orig_list[k]["cu_size_stg2"] = cu_size_stg2
                orig_list[k]["cu_size_stg3"] = cu_size_stg3
                orig_list[k]["cu_size_stg4"] = cu_size_stg4
                orig_list[k]["cu_size_stg5"] = cu_size_stg5
                orig_list[k]["cu_size_stg6"] = cu_size_stg6
                orig_list[k]["split_stg2"] = split_stg2
                orig_list[k]["split_stg3"] = split_stg3
                orig_list[k]["split_stg4"] = split_stg4
                orig_list[k]["split_stg5"] = split_stg5
                orig_list[k]["split"] = split_stg6
                orig_list[k]["real_CTU"] = real_CTU
                orig_list[k]["CTU"] = CTU

                # Update dataframe
                pd_row = pd.DataFrame({"POC": [CU_stg6["POC"]], "pic_name": CU_stg6["pic_name"], "cu_pos_x": CU_stg6["cu_pos"]["CU_loc_left"],\
                         "cu_pos_y": CU_stg6["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg6["cu_size"]["height"],\
                         "cu_size_w": CU_stg6["cu_size"]["width"], "split": CU_stg6["split"]})
                pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

                # Delete unnecessary entries
                for k2 in range(6):
                    del orig_list[k]["stg_"+str(k2+1)]

                del orig_list[k]["POC"]
                del orig_list[k]["pic_name"]

                # Save entry in final list
                mod_list.append(orig_list[k])

                utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)



def change_struct_16x16_no_dupl_v3(path_dir_l):
    """
        This version is like the change_struct_16x16_no_dupl_v2, but uses threads.
    """
    # Initial Steps
    print("Active function: change_struct_16x16_no_dupl_v3")
    t0 = time.time()

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_16x16_v3/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)
    #    pass

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_4"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    # Process files
    threads = []
    for f in files_l:
        x = threading.Thread(target=mod_16x16_threads, args=(f, path_dir_l, right_rows, columns, new_dir))
        threads.append(x)
        x.start()

    for thread in threads:
        thread.join()

    print("Time Elapsed: ", time.time()-t0)

def mod_16x16_threads(f, path_dir_l, right_rows, columns, new_dir):

    # Make labels path
    lbls_path = os.path.join(path_dir_l, f)

    # List to save entries
    mod_list = []

    # Read file
    orig_list = dataset_utils.file2lst(lbls_path[:-4])
    data_size = len(orig_list)
    #orig_frame = pd.DataFrame(orig_list)
    bool_list = list(map(right_rows, orig_list))
    idx_list = list(np.where(bool_list)[0])
    orig_list = list(map(orig_list.__getitem__, idx_list))

    # Dataframe initialization
    pd_full = pd.DataFrame(columns=columns)

    # Verbose
    print("Processing:", lbls_path)

    # Loop entries
    for k in range(len(orig_list)):
        
        # New entries
        CTU = orig_list[k]["stg_1"]
        CU_stg2 = orig_list[k]["stg_2"]
        CU_stg3 = orig_list[k]["stg_3"]
        CU_stg4 = orig_list[k]["stg_4"]
        
        # Verify size of the CU and if the variable structure is of what type
        if type(CU_stg4) == list or CU_stg4["cu_size"]["width"] != 16 or CU_stg4["cu_size"]["height"] != 16:
            continue

        # Verify if cu wasn't added already
        if  len(pd_full[(pd_full["POC"] ==  CU_stg4["POC"]) & (pd_full["pic_name"] == CU_stg4["pic_name"]) & \
            (pd_full["cu_size_w"] == CU_stg4["cu_size"]["width"]) & \
            (pd_full["cu_size_h"] == CU_stg4["cu_size"]["height"]) & \
            (pd_full["cu_pos_y"] == CU_stg4["cu_pos"]["CU_loc_top"]) &\
            (pd_full["cu_pos_x"] == CU_stg4["cu_pos"]["CU_loc_left"])]) == 0 : 
                
            cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
            cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"],CU_stg3["cu_size"]["width"]]), (1,-1))
            cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"],CU_stg4["cu_size"]["width"]]), (1,-1))
            RD0 = CU_stg4['RD0']
            RD1 = CU_stg4['RD1']
            RD2 = CU_stg4['RD2']
            RD3 = CU_stg4['RD3']
            RD4 = CU_stg4['RD4']
            RD5 = CU_stg4['RD5']
            RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
            split_stg2 = CU_stg2["split"]
            split_stg3 = CU_stg3["split"]
            split_stg4 = CU_stg4["split"]
            real_CTU = CTU["real_CTU"]

            # Add new entries
            orig_list[k]["RD"] = RD_tens
            orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
            orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
            orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
            orig_list[k]["cu_size_stg2"] = cu_size_stg2
            orig_list[k]["cu_size_stg3"] = cu_size_stg3
            orig_list[k]["cu_size_stg4"] = cu_size_stg4
            orig_list[k]["split_stg2"] = split_stg2
            orig_list[k]["split_stg3"] = split_stg3
            orig_list[k]["split"] = split_stg4
            orig_list[k]["real_CTU"] = real_CTU
            orig_list[k]["CTU"] = CTU

            # Update dataframe
            pd_row = pd.DataFrame({"POC": [CU_stg4["POC"]], "pic_name": CU_stg4["pic_name"], "cu_pos_x": CU_stg4["cu_pos"]["CU_loc_left"],\
                        "cu_pos_y": CU_stg4["cu_pos"]["CU_loc_top"], "cu_size_h": CU_stg4["cu_size"]["height"],\
                        "cu_size_w": CU_stg4["cu_size"]["width"], "split": CU_stg4["split"]})
            pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

            # Delete unnecessary entries
            for k2 in range(6):
                del orig_list[k]["stg_"+str(k2+1)]

            del orig_list[k]["POC"]
            del orig_list[k]["pic_name"]

            # Save entry in final list
            mod_list.append(orig_list[k])

            utils.echo("Complete: {per:.0%}".format(per=k/data_size))

    # Save list to file with the same name
    new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
    dataset_utils.lst2file(mod_list, new_path)

def change_struct_16x16(path_dir_l):
    """
        This version is meant to be used in to process the stage 4 data
    """
    print("Active function: change_struct_16x16")

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_16x16/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

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
            
            # New entries
            CTU = orig_list[k]["stg_1"]
            CU_stg2 = orig_list[k]["stg_2"]
            CU_stg3 = orig_list[k]["stg_3"]
            CU_stg4 = orig_list[k]["stg_4"]

            # Verify size of the CU and if the variable structure is of what type
            if type(CU_stg4) == list:
                continue

            elif CU_stg4["cu_size"]["width"] != 16 or CU_stg4["cu_size"]["height"] != 16:
                continue
                
            cu_pos_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_pos"]["CU_loc_top"] - CTU["cu_pos"]["CU_loc_top"],
                                   CU_stg2["cu_pos"]["CU_loc_left"] - CTU["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_pos_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_pos"]["CU_loc_top"] - CU_stg2["cu_pos"]["CU_loc_top"],
                                   CU_stg3["cu_pos"]["CU_loc_left"] - CU_stg2["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_pos_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_pos"]["CU_loc_top"] - CU_stg3["cu_pos"]["CU_loc_top"],
                                   CU_stg4["cu_pos"]["CU_loc_left"] - CU_stg3["cu_pos"]["CU_loc_left"]]), (1,-1))
            cu_size_stg2 = torch.reshape(torch.tensor([CU_stg2["cu_size"]["height"], CU_stg2["cu_size"]["width"]]), (1,-1))
            cu_size_stg3 = torch.reshape(torch.tensor([CU_stg3["cu_size"]["height"], CU_stg3["cu_size"]["width"]]), (1,-1))
            cu_size_stg4 = torch.reshape(torch.tensor([CU_stg4["cu_size"]["height"], CU_stg4["cu_size"]["width"]]), (1,-1))

            RD0 = CU_stg4['RD0']
            RD1 = CU_stg4['RD1']
            RD2 = CU_stg4['RD2']
            RD3 = CU_stg4['RD3']
            RD4 = CU_stg4['RD4']
            RD5 = CU_stg4['RD5']
            RD_tens = torch.reshape(torch.tensor([[RD0, RD1, RD2, RD3, RD4, RD5]]), (1, -1))
            split_stg2 = CU_stg2["split"]
            split_stg3 = CU_stg3["split"]
            split_stg4 = CU_stg4["split"]
            real_CTU = CTU["real_CTU"]

            # Add new entries
            orig_list[k]["RD"] = RD_tens
            orig_list[k]["cu_pos_stg2"] = cu_pos_stg2
            orig_list[k]["cu_pos_stg3"] = cu_pos_stg3
            orig_list[k]["cu_pos_stg4"] = cu_pos_stg4
            orig_list[k]["cu_size_stg2"] = cu_size_stg2
            orig_list[k]["cu_size_stg3"] = cu_size_stg3
            orig_list[k]["cu_size_stg4"] = cu_size_stg4
            orig_list[k]["split_stg2"] = split_stg2
            orig_list[k]["split_stg3"] = split_stg3
            orig_list[k]["split"] = split_stg4
            orig_list[k]["real_CTU"] = real_CTU
            orig_list[k]["CTU"] = CTU

            # Delete unnecessary entries
            for k2 in range(6):
                del orig_list[k]["stg_"+str(k2+1)]

            del orig_list[k]["POC"]
            del orig_list[k]["pic_name"]

            # Save entry in final list
            mod_list.append(orig_list[k])

            utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4])
        dataset_utils.lst2file(mod_list, new_path)

def change_struct_no_dupl_stg_4_complexity_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stages 4. Here it is going to be obtained data to be used for the complexity assesment
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg_4_complexity_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg_4_compl_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_4']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_4"]['RD0'], l["stg_4"]['RD1'], l["stg_4"]['RD2'], l["stg_4"]['RD3'], l["stg_4"]['RD4'], l["stg_4"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_3"]["cu_pos"]["CU_loc_top"] - l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"] - l["stg_2"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_4"]["cu_pos"]["CU_loc_top"] - l["stg_3"]["cu_pos"]["CU_loc_top"], l["stg_4"]["cu_pos"]["CU_loc_left"] - l["stg_3"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                               [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]], \
                               [l["stg_4"]["cu_size"]["height"], l["stg_4"]["cu_size"]["width"]], \
                                l["stg_2"]["split"], l["stg_3"]["split"], l["stg_4"]["split"], \
                                l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"], \
                                l["POC"], l["pic_name"], \
                                [l["stg_4"]["cu_pos"]["CU_loc_top"], l["stg_4"]["cu_pos"]["CU_loc_left"]], \
                                [l["stg_4"]["cu_size"]["height"], l["stg_4"]["cu_size"]["width"]] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg_2_complexity_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stages 2. Here it is going to be obtained data to be used for the complexity assesment
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg_2_complexity_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg_4_compl_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_2']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_2"]['RD0'], l["stg_2"]['RD1'], l["stg_2"]['RD2'], l["stg_2"]['RD3'], l["stg_2"]['RD4'], l["stg_2"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                                l["stg_2"]["split"], \
                                l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"], \
                                l["POC"], l["pic_name"], \
                                [l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"]], \
                                [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)

def change_struct_no_dupl_stg_6_complexity_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stages 6. Here it is going to be obtained data to be used for the complexity assesment
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg_6_complexity_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg_6_compl_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_6']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_6"]['RD0'], l["stg_6"]['RD1'], l["stg_6"]['RD2'], l["stg_6"]['RD3'], l["stg_6"]['RD4'], l["stg_6"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_3"]["cu_pos"]["CU_loc_top"] - l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"] - l["stg_2"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_4"]["cu_pos"]["CU_loc_top"] - l["stg_3"]["cu_pos"]["CU_loc_top"], l["stg_4"]["cu_pos"]["CU_loc_left"] - l["stg_3"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_5"]["cu_pos"]["CU_loc_top"] - l["stg_4"]["cu_pos"]["CU_loc_top"], l["stg_5"]["cu_pos"]["CU_loc_left"] - l["stg_4"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_6"]["cu_pos"]["CU_loc_top"] - l["stg_5"]["cu_pos"]["CU_loc_top"], l["stg_6"]["cu_pos"]["CU_loc_left"] - l["stg_5"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                               [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]], \
                               [l["stg_4"]["cu_size"]["height"], l["stg_4"]["cu_size"]["width"]], \
                               [l["stg_5"]["cu_size"]["height"], l["stg_5"]["cu_size"]["width"]], \
                               [l["stg_6"]["cu_size"]["height"], l["stg_6"]["cu_size"]["width"]], \
                                l["stg_2"]["split"], l["stg_3"]["split"], l["stg_4"]["split"], l["stg_5"]["split"], l["stg_6"]["split"], \
                                l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"], \
                                l["POC"], l["pic_name"], \
                                [l["stg_6"]["cu_pos"]["CU_loc_top"], l["stg_6"]["cu_pos"]["CU_loc_left"]], \
                                [l["stg_6"]["cu_size"]["height"], l["stg_6"]["cu_size"]["width"]] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg_5_complexity_v4(path_dir_l):
    """
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stages 5. Here it is going to be obtained data to be used for the complexity assesment
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg_5_complexity_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg_5_compl_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = dataset_utils.get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = dataset_utils.file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_5']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_5"]['RD0'], l["stg_5"]['RD1'], l["stg_5"]['RD2'], l["stg_5"]['RD3'], l["stg_5"]['RD4'], l["stg_5"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_3"]["cu_pos"]["CU_loc_top"] - l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"] - l["stg_2"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_4"]["cu_pos"]["CU_loc_top"] - l["stg_3"]["cu_pos"]["CU_loc_top"], l["stg_4"]["cu_pos"]["CU_loc_left"] - l["stg_3"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_5"]["cu_pos"]["CU_loc_top"] - l["stg_4"]["cu_pos"]["CU_loc_top"], l["stg_5"]["cu_pos"]["CU_loc_left"] - l["stg_4"]["cu_pos"]["CU_loc_left"]], \
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                               [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]], \
                               [l["stg_4"]["cu_size"]["height"], l["stg_4"]["cu_size"]["width"]], \
                               [l["stg_5"]["cu_size"]["height"], l["stg_5"]["cu_size"]["width"]], \
                                l["stg_2"]["split"], l["stg_3"]["split"], l["stg_4"]["split"], l["stg_5"]["split"], \
                                l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"], \
                                l["POC"], l["pic_name"], \
                                [l["stg_5"]["cu_pos"]["CU_loc_top"], l["stg_5"]["cu_pos"]["CU_loc_left"]], \
                                [l["stg_5"]["cu_size"]["height"], l["stg_5"]["cu_size"]["width"]] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        dataset_utils.lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


# Main Function
def main():
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels/mod_with_real_CTU/"  # Path with the labels processed
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU"  # Path with the labels processed
    path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels/mod_with_real_CTU/complexity"  # For training Labels path
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels_for_testing/test"  # Path with the labels processed
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_stg4_v4/train_valid_test/balanced_labels_downsamp/test"

    print("Modifying struct from:", path_dir_l)

    change_struct_no_dupl_stg_6_complexity_v4(path_dir_l)
    #change_struct_no_dupl_stg2_v4(path_dir_l)
    #change_struct_8x8_no_dupl_v2(path_dir_l)
    #change_struct_32x16_no_dupl_v2(path_dir_l)
    # change_struct_no_dupl_stg5_v4(path_dir_l)
    # change_struct_no_dupl_stg6_v4(path_dir_l)
    #change_struct_32x32_no_dupl_v3(path_dir_l)
    #change_struct_32x32(path_dir_l)
    #change_struct_32x32_no_dupl_v2(path_dir_l)
    #change_struct_64x64(path_dir_l)
    #change_struct_32x32_eval(path_dir_l)

if __name__ == "__main__":
    main()
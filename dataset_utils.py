"""@package docstring 

@file dataset_utils.py 

@brief Usefull functions to manipulate data, change and create structures 
 
@section libraries_dataset_utils Libraries 
- os
- utils
- pandas
- torch
- csv
- struct
- numpy
- sklearn.model_selection
- cv2
- threading
- pickle
- shutil
- sys
- time
- math
- re

@section classes_dataset_utils Classes 
- VideoCaptureYUV
 
@section functions_dataset_utils Functions
- bgr2yuv(matrix)
- extract_content(f)
- file_stats(path)
- show_bin_content(path, num_records=100)
- add_best_split(labels)
- read_from_records(path, num_records)
- process_info(content)
- match_cu(CU, CTU, position, size)
- find_cu(df_cu, CTU, position, size)
- build_entry(stg1=[], stg2=[], stg3=[], stg4=[], stg5=[], stg6=[])
- add_cu_to_dict(cu_dict, cu)
- transform_create_struct_faster_v2_mod_divs(f, f_name, num_records, output_dir, n_output_file, color_ch=0)
- transform_create_struct_faster_v3(f, f_name, num_records, output_dir, n_output_file, color_ch=0)
- process_ctus_cus(df_ctus, df_cus)
- split(size, pos, split_mode)
- transform_raw_dataset(dic)
- get_files_from_folder(path, endswith=".yuv")
- get_num_frames(path, name, width, height)
- get_file_metadata_info(path, name)
- get_file_metadata_info_mod(name)
- encode_dataset
- compute_split_per_depth(d_path)
- compute_split_per_depth_v2(d_path)
- compute_split_per_depth_v3(d_path)
- lst2csv(lst, name_of_file)
- get_some_data_equaly(X, path_dir_l, classes, split_pos)
- lst2csv_v2(lst_lst, n_file, n_fields)
- csv2lst(csv_file)
- file2lst(file)
- lst2file(lst, name_of_file)
- unite_labels_v6(dir_path_l, n_output_file="labels_pickle", color_ch=0)
- unite_labels_v6_mod(dir_path_l, n_output_file="labels_pickle", color_ch=0)
- create_dir(output_dir)
- labels_with_specific_cch(dir_path, cch=0)
- read_from_records_v2(f, f_name, num_records)
- file_stats_v2(path)
- compute_split_proportions(path, num_cus=float('inf'))
- compute_split_proportions_with_custom_data(custom_dataset, stage, num_cus=float('inf'))
- compute_split_proportions_with_custom_data_multi(custom_dataset, split_pos_in_struct, num_cus=float('inf'))
- compute_split_proportions_with_path_multi_new(path, split_pos_in_struct, num_cus=float('inf'))
- compute_split_proportions_with_custom_data_multi_new(custom_dataset, split_pos_in_struct, num_cus=float('inf'))
- compute_split_proportions_labels(path, num_cus=float('inf'))
- balance_dataset(dir_path, stg, n_classes=6)
- balance_dataset_JF(dir_path, n_classes=6)
- balance_dataset_down(dir_path, n_classes=6)
- balance_dataset_down_v2(dir_path)
- balance_dataset_down_v3(dir_path)
- balance_dataset_down_v4(dir_path)
- balance_dataset_up(dir_path, n_classes=6)
- balance_dataset_up_v2(dir_path)
- balance_dataset_up_v3(dir_path)
- gen_dataset_types(d_path, valid_percent)
- change_struct_64x64_eval(path_dir_l)
- change_struct_32x32_eval(path_dir_l)
- change_struct_64x64(path_dir_l)
- change_struct_64x64_no_dupl_v3(path_dir_l)
- mod_64x64_threads(f, path_dir_l, right_rows, columns, new_dir)
- change_struct_64x64_no_dupl_v2(path_dir_l)
- change_struct_32x32(path_dir_l)
- change_struct_32x32_no_dupl(path_dir_l)
- change_struct_32x32_no_dupl_v2(path_dir_l)
- change_struct_32x32_no_dupl_v3(path_dir_l)
- mod_32x32_threads(f, path_dir_l, right_rows, columns, new_dir)
- change_struct_32x32_no_dupl_v2_test(path_dir_l)
- change_struct_16x16_no_dupl_v2(path_dir_l)
- list2tuple(l)
- tuple2list(l)
- change_struct_8x8_no_dupl_v2(path_dir_l)
- change_struct_no_dupl_stg6_v4(path_dir_l)
- change_struct_no_dupl_stg5_v4(path_dir_l)
- change_struct_no_dupl_stg2_v4(path_dir_l)
- change_struct_no_dupl_stg4_v4(path_dir_l)
- change_struct_no_dupl_stg3_v4(path_dir_l)
- change_struct_32x16_no_dupl_v2(path_dir_l)
- change_struct_32x8_no_dupl_v2(path_dir_l)
- change_struct_16x8_no_dupl_v2(path_dir_l)
- change_struct_8x4_no_dupl_v2(path_dir_l)
- change_struct_32x4_no_dupl_v2(path_dir_l)
- change_struct_16x4_no_dupl_v2(path_dir_l)
- change_struct_16x16_no_dupl_v3(path_dir_l)
- mod_16x16_threads(f, path_dir_l, right_rows, columns, new_dir)
- change_struct_16x16(path_dir_l)
- change_struct_no_dupl_stg_4_complexity_v4(path_dir_l)
- change_struct_no_dupl_stg_3_complexity_v4(path_dir_l)
- change_struct_no_dupl_stg_2_complexity_v4(path_dir_l)
- change_struct_no_dupl_stg_6_complexity_v4(path_dir_l)
- change_struct_no_dupl_stg_5_complexity_v4(path_dir_l)
 
@section global_vars_dataset_utils Global Variables 
- None 

@section todo_dataset_utils TODO 
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

@section author_dataset_utils Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:22:04.120175
"""

# ==============================================================
# Imports
# ==============================================================

import os
import math
import struct
import re
import time
import cv2
import numpy as np
import csv
import pandas as pd
import pickle
import shutil
import utils
import torch
from sklearn.model_selection import train_test_split
import threading

# ==============================================================
# Classes
# ==============================================================

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.y_len = self.width * self.height
        self.frame_len = int(self.y_len * 3 / 2)
        self.f = open(filename, 'rb')
        self.shape = (int(self.height), int(self.width))

    def read_raw(self, frame_num=None):
        try:
            if frame_num:
                self.f.seek(0)
                self.f.seek(frame_num * self.frame_len)
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            y = yuv[:self.y_len].reshape(self.shape)
            uv = yuv[self.y_len:].reshape((int(self.height), int(self.width / 2)))
            yuv = np.concatenate((y, uv), axis=1)
            u = uv[:int(self.height/2), :]
            v = uv[int(self.height/2):, :]

        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv, y, u, v

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        # bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)#cv2.COLOR_YUV2BGR_NV21)
        return ret, yuv


# ==============================================================
# Functions
# ==============================================================

def bgr2yuv(matrix):
    """!
    @brief Converts BGR matrix to YUV matrix

    @param [in] matrix: BGR matrix
    @param [out] YUV: YUV conversion
    """

    # Convert from bgr to yuv
    YUV = cv2.cvtColor(matrix, cv2.COLOR_BGR2YUV_I420)  # cv2.COLOR_YUV2BGR_NV21)

    return YUV

def extract_content(f):
    """!
    @brief Extract a single record from binary file

    @param [in] f: File object
    @param [out] content: Dictionary containing the information of a single record
    """

    # Get all the 16 bit (2 bytes) information and add info to dict
    info = f.read(2)  # Read 2 bytes from file
    POC = struct.unpack("H", info)[0]  # Picture order count

    info = f.read(2)  # Read 2 bytes from file
    color_ch = struct.unpack("H", info)[0]  # Color channel

    info = f.read(2)  # Read 2 bytes from file
    CU_loc_left = struct.unpack("H", info)[0]  # CU location left

    info = f.read(2)  # Read 2 bytes from file
    CU_loc_top = struct.unpack("H", info)[0]  # CU location top

    info = f.read(2)  # Read 2 bytes from file
    CU_w = struct.unpack("H", info)[0]  # CU width

    info = f.read(2)  # Read 2 bytes from file
    CU_h = struct.unpack("H", info)[0]  # CU height

    # Get all the 64 bit (8 bytes) information and add info to dict
    info = f.read(8)  # Read 8 bytes from file
    RD_mode0 = struct.unpack("d", info)[0]  # RD of the mode 0

    info = f.read(8)  # Read 8 bytes from file
    RD_mode1 = struct.unpack("d", info)[0]  # RD of the mode 1

    info = f.read(8)  # Read 8 bytes from file
    RD_mode2 = struct.unpack("d", info)[0]  # RD of the mode 2

    info = f.read(8)  # Read 8 bytes from file
    RD_mode3 = struct.unpack("d", info)[0]  # RD of the mode 3

    info = f.read(8)  # Read 8 bytes from file
    RD_mode4 = struct.unpack("d", info)[0]  # RD of the mode 4

    info = f.read(8)  # Read 8 bytes from file
    RD_mode5 = struct.unpack("d", info)[0]  # RD of the mode 5

    content = {"POC": POC, "color_ch": color_ch, "CU_loc_left": CU_loc_left,
               "CU_loc_top": CU_loc_top, "CU_w": CU_w, "CU_h": CU_h, "RD_mode0": RD_mode0,
               "RD_mode1": RD_mode1, "RD_mode2": RD_mode2, "RD_mode3": RD_mode3,
               "RD_mode4": RD_mode4, "RD_mode5": RD_mode5}  # Dictionary with record

    return content

def file_stats(path):
    """!
    @brief Finds out the size of the binary file and computes the number of records

    @param [in] path: Path where the binary file is located
    @param [out] num_records: Number of records that the binary file contains
    @param [out] file_size: Size of the binary file
    """

    print("Obtaining information from file ", path, "...")

    # Obtaining number of records
    file_size = os.path.getsize(path)  # Get file size
    print("File Size is :", file_size, "bytes")
    num_records = math.ceil(file_size / 60)  # Each record as 60 bytes
    print("Number of records:", num_records)
    print()

    return num_records, file_size

def show_bin_content(path, num_records=100):
    """!
    @brief Show contents of a binary file containing encoding information

    @param [in] path: Path where the binary file is located
    @param [in] num_records: Number of records to show
    """

    # Reading and extracting information from file
    f = open(path, "rb")  # Opening file in binary mode read and write

    for l in range(0, num_records):
        print("Record", l)
        # Extract Picture order, color channel, CU locations, CU width, CU height
        print("2 Byte information / uint16_t")
        for p in range(0, 6):
            info = f.read(2)

            print(struct.unpack("H", info)[0])

        print("8 Byte information / double type")
        for p in range(0, 6):
            info = f.read(8)

            try:
                print(struct.unpack("d", info)[0])
            except:
                print(info)
                raise Exception("Something went wrong!")

        print()

    f.close()

def add_best_split(labels):
    """!
    @brief Modifies labels by adding an extra parameter

    Dictionary containing all the info about the file: It's a dictionary of picture numbers, which then leads to a
    dictionary of the info. For example: records = {"Pic_0" :{"CU_0": {"colorChannel": 1,
                                               "CULoc_left": 2,
                                                ...
                                                "split": 5
                                             }
                                     ...
                            ...      }
                            }

    @param [in] labels: Dictionary with the labels of the dataset
    @param [out] new_labels: New dictionary with the lables of the dataset
    """

    new_labels = labels.copy()

    for pic_key in new_labels.keys():

        for CU_key in new_labels[pic_key].keys():
            RDs_lst = []  # List that contains RD values for diferent modes

            for p in range(0, 6):
                # Obtain RD mode value
                rd = new_labels[pic_key][CU_key]["RD_mode" + str(p)]
                if rd == 0.0:  # If RD is 0.0, then it means the RD value is not valid
                    rd = math.inf
                RDs_lst.append(rd)  # Add RD values of each mode

            best_mode = RDs_lst.index(min(RDs_lst))  # Obtain best mode to split
            new_labels[pic_key][CU_key]["split"] = best_mode  # Add to dictionary

    return new_labels

def read_from_records(path, num_records):
    """!
    @brief Read the information/file generated by the encoder
    Dictionary containing all the info about the file: It's a dictionary of picture numbers, which then leads to a
    dictionary of the info. For example: records = {"Pic_0" :{"CU_0": {"colorChannel": 1,
                                               "CULoc_left": 2,
                                                ...
                                             }
                                     ...
                            ...      }
                            }

    @param [in] path: Path where the file is located
    @param [in] num_records: Number of records to show
    @param [out] records: Dictionary containing the information of all records
    """

    records = {}  # Dictionary with all the information extracted from the file
    CU_Counter = {}  # Dictionary keeping track of the number of CUs per Picture

    # Reading and extracting information from file
    f = open(path, "rb")  # Opening file in binary mode read and write
    f_name = path.split("\\")[-1][:-4]  # Get File name

    # Extract records
    for i in range(0, num_records):
        content = extract_content(f)
        pic_key = f_name + "_Pic_" + str(content["POC"])  # Key for the records and CU_counter dictionary
        # Update CU counter
        try:
            CU_Counter[pic_key] = CU_Counter[pic_key] + 1  # In case pic_key is not yet in the dictionary...
        except:
            # ... add it
            CU_Counter[pic_key] = 1

        CU_key = "CU_" + str(CU_Counter[pic_key] - 1)  # Key for the records dictionary
        # Store record
        try:
            # In case CU_key is not yet associated with a dictionary inside of the dictionary records...
            records[pic_key][CU_key] = {"POC": content["POC"], "color_ch": content["color_ch"],
                                        "CU_loc_left": content["CU_loc_left"],
                                        "CU_loc_top": content["CU_loc_top"], "CU_w": content["CU_w"],
                                        "CU_h": content["CU_h"],
                                        "RD_mode0": content["RD_mode0"], "RD_mode1": content["RD_mode1"],
                                        "RD_mode2": content["RD_mode2"], "RD_mode3": content["RD_mode3"],
                                        "RD_mode4": content["RD_mode4"],
                                        "RD_mode5": content["RD_mode5"]}

        except:
            # ... Associate pic_key with a dictionary, and then add CU_key and item with the dictionary created
            records[pic_key] = {}
            records[pic_key][CU_key] = {"POC": content["POC"], "color_ch": content["color_ch"],
                                        "CU_loc_left": content["CU_loc_left"],
                                        "CU_loc_top": content["CU_loc_top"], "CU_w": content["CU_w"],
                                        "CU_h": content["CU_h"],
                                        "RD_mode0": content["RD_mode0"], "RD_mode1": content["RD_mode1"],
                                        "RD_mode2": content["RD_mode2"], "RD_mode3": content["RD_mode3"],
                                        "RD_mode4": content["RD_mode4"],
                                        "RD_mode5": content["RD_mode5"]}
    f.close()  # Detach file

    return records, CU_Counter

def process_info(content):
    """!
    @brief Process the raw data from the labels given by the encoder

    @param [in] content: Dict with the information about
    @param [out] content: Processed dict
    """

    # Add Best split
    RDs_lst = []
    for p in range(0, 6):
        # Obtain RD mode value
        rd = content["RD_mode" + str(p)]
        if rd == 0.0:  # If RD is 0.0, then it means the RD value is not valid
            rd = math.inf
        RDs_lst.append(rd)  # Add RD values of each mode

    best_mode = RDs_lst.index(min(RDs_lst))  # Obtain best mode to split
    content["split"] = best_mode  # Add to dictionary
    content["pic_name"] = content["f_name_label"][:-4] # Add to dictionary
    content["cu_size"] = {"height": content["CU_h"], "width": content["CU_w"]}
    content["cu_pos"] = {"CU_loc_left": content["CU_loc_left"], "CU_loc_top": content["CU_loc_top"]}
    del content["CU_w"]
    del content["CU_h"]
    del content["CU_loc_left"]
    del content["CU_loc_top"]

    # Change name of RDs keys
    for p in range(0, 6):
        content["RD" + str(p)] = content["RD_mode" + str(p)]
        del content["RD_mode" + str(p)]

    return content

def match_cu(CU, CTU, position, size):
    """!
    @brief Verifies if the CUs are the same based in their position, size and other information

    @param [in] CU: CU (dict with information about the CU) that will be inspected
    @param [in] CTU: Original CTU (dict with information about the CTU)
    @param [in] position: Position of the CU that it is being searched
    @param [in] size: Position of the CU that it is being searched
    @param [out] match_or_not: Bool value with the decision about the matching
    """

    # Verify CU file
    cond_POC_fname = CTU["POC"] == CU["POC"] and CU["f_name_label"] == CU["f_name_label"]

    # Verify CU position
    cu_pos = (CU["cu_pos"]['CU_loc_left'], CU["cu_pos"]['CU_loc_top'])
    cond_same_pos = cu_pos == position

    # Verify color channel
    cond_colorch = CU["color_ch"] == CTU["color_ch"]

    # Verify size
    cu_size = (CU["cu_size"]['width'], CU["cu_size"]['height'])
    cond_size = cu_size == size

    return cond_POC_fname and cond_same_pos and cond_colorch and cond_size

def find_cu(df_cu, CTU, position, size):
    """!
    @brief Verifies if the CU is in the dataframe, using the size and other information. Uses pandas' dataframe

    @param [in] df_cu: Dataframe with all the CUs
    @param [in] CTU: Original CTU (dict with information about the CTU)
    @param [in] position: Position of the CU that it is being searched [left, top]
    @param [in] size: Position of the CU that it is being searched [left, top]
    @param [out] cu: Either a CU pandas' series object or a false boolean value that indicates that the CU wasn't found
    """

    cu = df_cu[(df_cu["POC"] == CTU["POC"]) & (df_cu["color_ch"] == CTU["color_ch"]) & (df_cu["f_name_label"] == CTU["f_name_label"]) &
               (df_cu["CU_loc_left"] == position[0]) & (df_cu["CU_loc_top"] == position[1]) &
               (df_cu["width"] == size[0]) & (df_cu["height"] == size[1])]

    # Return false in case the CU isn't found
    if len(cu) == 0:
        cu = False

    return cu

def build_entry(stg1=[], stg2=[], stg3=[], stg4=[], stg5=[], stg6=[]):
    """!
    @brief Builds a entry with all information needed for each stage, and also removes unnecessary info

    @param [in] stg1: CU (dict with information about the CU) for stage 1
    @param [in] stg2: CU (dict with information about the CU) for stage 2
    @param [in] stg3: CU (dict with information about the CU) for stage 3
    @param [in] stg4: CU (dict with information about the CU) for stage 4
    @param [in] stg5: CU (dict with information about the CU) for stage 5
    @param [in] stg6: CU (dict with information about the CU) for stage 6
    @param [out] entry: Dictionary with information about the all stages inputs
    """

    # Save POC and pic name information
    POC = stg1["POC"]
    pic_name = stg1["pic_name"]

    # Build entry
    entry = {"POC": POC, "pic_name": pic_name, "stg_1": stg1, "stg_2": stg2, "stg_3": stg3, "stg_4": stg4, "stg_5": stg5,
                         "stg_6": stg6}
    return entry

def add_cu_to_dict(cu_dict, cu):
    """!
    @brief Adds information of a specific CU to the dictionary

    @param [in] cu_dict: Dictionary with information about all CUs
    @param [in] cu: CU information to add to the dictionary
    @param [out] cu_dict: Dictionary with information about all CUs, with a new cu added
    """

    for key in cu_dict.keys():
        if key == "width" or key == "height":
            cu_dict[key].append(cu["cu_size"][key])  # Add CU information for cu_size

        elif key == "CU_loc_left" or key == "CU_loc_top":
            cu_dict[key].append(cu["cu_pos"][key])  # Add CU information for cu_pos

        elif key == "all_info":
            cu_dict[key].append(cu)  # Add entire CU information to the dict

        else:
            cu_dict[key].append(cu[key])  # Add any other information to it's specific key

    return cu_dict

def transform_create_struct_faster_v2_mod_divs(f, f_name, num_records, output_dir, n_output_file, color_ch=0):
    """!
    @brief First obtains all CTUs and CUs in the file using a dictionary/dataframe, afterward organizes them in a stage
           oriented way. Removes elements from the cu list to speed up the process. Uses only specified color
           channel. This versions divides info into multiple files

    @param [in] f: File object
    @param [in] f_name: File name
    @param [in] num_records: Number of records
    @param [in] color_ch: Color channel
    @param [out] structed_cus: Dictionary containing the all CUs organized in a stage oriented way. Each entry looks
    like: [f_name_labels, pic_name, RD0, RD1, RD2, RD3, RD4, RD5, pos, size]
    """
    # Initialize Variables
    # Dictionary with that will contain all data
    CUs_dict = {"POC": [], "color_ch": [], "f_name_label": [], "split": [], "pic_name": [], "width": [], "height": [],
                "CU_loc_left": [], "CU_loc_top": [], "RD0": [], "RD1": [], "RD2": [], "RD3": [], "RD4": [], "RD5": [],
                "all_info": []}
    CTUs_dict = {"POC": [], "color_ch": [], "f_name_label": [], "split": [], "pic_name": [], "width": [], "height": [],
                "CU_loc_left": [], "CU_loc_top": [], "RD0": [], "RD1": [], "RD2": [], "RD3": [], "RD4": [], "RD5": [],
                "all_info": []}
    # Initizalize other variables
    count = 0
    accumulate_data = 10  # Can be changed, it represents the number of frames that will be included
    POC_counter = 0
    prcss_count = 0

    # Log
    print("Completing processing", f_name)

    # Extract records
    for i in range(0, num_records):
        content = extract_content(f)

        POC = content["POC"]
        if POC_counter != POC:  # Mechanism to verify that the POC has changed
            POC_counter += 1
            prcss_count += 1

        if prcss_count == accumulate_data:
            # Reset variable
            prcss_count = 0
            # Convert dict to dataframe
            df_cus = pd.DataFrame(data=CUs_dict)
            df_ctus = pd.DataFrame(data=CTUs_dict)

            # Process data
            structed_cus = process_ctus_cus(df_ctus, df_cus)

            # Write file
            lst2file(structed_cus, output_dir + "/" + f_name +"_" + n_output_file + "_" + str(count))
            print("New file output!")

            # Update counter
            count += 1

            # Reset the dictionaries
            CUs_dict = {"POC": [], "color_ch": [], "f_name_label": [], "split": [], "pic_name": [], "width": [],
                        "height": [],
                        "CU_loc_left": [], "CU_loc_top": [], "RD0": [], "RD1": [], "RD2": [], "RD3": [], "RD4": [],
                        "RD5": [],
                        "all_info": []}
            CTUs_dict = {"POC": [], "color_ch": [], "f_name_label": [], "split": [], "pic_name": [], "width": [],
                         "height": [],
                         "CU_loc_left": [], "CU_loc_top": [], "RD0": [], "RD1": [], "RD2": [], "RD3": [], "RD4": [],
                         "RD5": [],
                         "all_info": []}

        # Add CU to dictionary
        if content["CU_w"] == 128 and content["CU_h"] == 128 and content["color_ch"] == color_ch:
            content["f_name_label"] = f_name
            content = process_info(content)  # Adds the best split and modifies the dictionary
            CTUs_dict = add_cu_to_dict(CTUs_dict, content)

        elif content["color_ch"] == color_ch:
            content["f_name_label"] = f_name
            content = process_info(content)  # Adds the best split and modifies the dictionary
            CUs_dict = add_cu_to_dict(CUs_dict, content)


    if len(CTUs_dict["POC"]) != 0:
        # Convert dict to dataframe
        df_cus = pd.DataFrame(data=CUs_dict)
        df_ctus = pd.DataFrame(data=CTUs_dict)

        # Process data
        structed_cus = process_ctus_cus(df_ctus, df_cus)

        # Write file
        lst2file(structed_cus, output_dir + "/" + f_name + "_" + n_output_file + "_" + str(count))
        print("New file output!")

    f.close()

def transform_create_struct_faster_v3(f, f_name, num_records, output_dir, n_output_file, color_ch=0):
    """!
    @brief First obtains all CTUs and CUs in the file using a dictionary/dataframe, afterward organizes them in a stage
           oriented way. Removes elements from the cu list to speed up the process. Uses only specified color
           channel. This version its similar to the div version, but outputs only a file

    @param [in] f: File object
    @param [in] f_name: File name
    @param [in] num_records: Number of records
    @param [in] color_ch: Color channel
    @param [out] structed_cus: Dictionary containing the all CUs organized in a stage oriented way. Each entry looks
    like: [f_name_labels, pic_name, RD0, RD1, RD2, RD3, RD4, RD5, pos, size]
    """
    # Initialize Variables
    # Dictionary with that will contain all data
    CUs_dict = {"POC": [], "color_ch": [], "f_name_label": [], "split": [], "pic_name": [], "width": [], "height": [],
                "CU_loc_left": [], "CU_loc_top": [], "RD0": [], "RD1": [], "RD2": [], "RD3": [], "RD4": [], "RD5": [],
                "all_info": []}
    CTUs_dict = {"POC": [], "color_ch": [], "f_name_label": [], "split": [], "pic_name": [], "width": [], "height": [],
                "CU_loc_left": [], "CU_loc_top": [], "RD0": [], "RD1": [], "RD2": [], "RD3": [], "RD4": [], "RD5": [],
                "all_info": []}
    # Initizalize other variables
    accumulate_data = 1
    POC_counter = 0
    prcss_count = 0
    struct_cus_final = []

    # Log
    print("Completing processing", f_name)

    # Extract records
    for i in range(0, num_records):
        content = extract_content(f)

        POC = content["POC"]
        if POC_counter != POC:  # Mechanism to verify that the POC has changed
            POC_counter += 1
            prcss_count += 1

        if prcss_count == accumulate_data:
            # Reset variable
            prcss_count = 0
            # Convert dict to dataframe
            df_cus = pd.DataFrame(data=CUs_dict)
            df_ctus = pd.DataFrame(data=CTUs_dict)

            # Process data
            structed_cus = process_ctus_cus(df_ctus, df_cus)
            struct_cus_final.extend(structed_cus)

            # Reset the dictionaries
            CUs_dict = {"POC": [], "color_ch": [], "f_name_label": [], "split": [], "pic_name": [], "width": [],
                        "height": [],
                        "CU_loc_left": [], "CU_loc_top": [], "RD0": [], "RD1": [], "RD2": [], "RD3": [], "RD4": [],
                        "RD5": [],
                        "all_info": []}
            CTUs_dict = {"POC": [], "color_ch": [], "f_name_label": [], "split": [], "pic_name": [], "width": [],
                         "height": [],
                         "CU_loc_left": [], "CU_loc_top": [], "RD0": [], "RD1": [], "RD2": [], "RD3": [], "RD4": [],
                         "RD5": [],
                         "all_info": []}

        # Add CU to dictionary
        if content["CU_w"] == 128 and content["CU_h"] == 128 and content["color_ch"] == color_ch:
            content["f_name_label"] = f_name
            content = process_info(content)  # Adds the best split and modifies the dictionary
            CTUs_dict = add_cu_to_dict(CTUs_dict, content)

        elif content["color_ch"] == color_ch:
            content["f_name_label"] = f_name
            content = process_info(content)  # Adds the best split and modifies the dictionary
            CUs_dict = add_cu_to_dict(CUs_dict, content)


    if len(CTUs_dict["POC"]) != 0:
        # Convert dict to dataframe
        df_cus = pd.DataFrame(data=CUs_dict)
        df_ctus = pd.DataFrame(data=CTUs_dict)

        # Process data
        structed_cus = process_ctus_cus(df_ctus, df_cus)
        struct_cus_final.extend(structed_cus)

    lst2file(struct_cus_final, output_dir + "/" + f_name + "_" + n_output_file)
    print("New file output!")
    f.close()

def process_ctus_cus(df_ctus, df_cus):
    """!
    @brief Function to create data structures to organize the CTUs and CUs

    @param [in] df_ctus: Dataframe with CTUs
    @param [in] df_cus: Dataframe with CUs
    @param [out] structed_cus: Dictionary containing the all CUs organized in a stage oriented way. Each entry looks
    like: [f_name_labels, pic_name, RD0, RD1, RD2, RD3, RD4, RD5, pos, size]
    """

    # Total amount of CTUs
    total_amount_ctus = len(df_ctus)

    structed_cus = []
    for index, row in df_ctus.iterrows():

        if index % 5 == 0:
            utils.echo("* {percentage:.0%}".format(percentage=index / total_amount_ctus))

        CTU = row

        # Stage 1
        # Get CTU Information
        ctu_size = (CTU["width"], CTU["height"])
        ctu_pos = (CTU["CU_loc_left"], CTU["CU_loc_top"])

        # Split CTU
        positions_stg2, sizes_stg2 = split(ctu_size, ctu_pos, 1)

        # Stage 2
        for p1 in range(len(positions_stg2)):
            # Get CU
            cu_stg2 = find_cu(df_cus, CTU, positions_stg2[p1], sizes_stg2[p1])

            if type(cu_stg2) != bool:  # If not found, it will be ignored

                # Stop search
                if cu_stg2["split"].item() == 0:
                    # Build dic until this point and add line to list
                    structed_cus.append(build_entry(CTU["all_info"], cu_stg2["all_info"].item()))

                else:
                    # Split the current CU
                    cu_size_stg2 = (cu_stg2["width"].item(), cu_stg2["height"].item())
                    cu_pos_stg2 = (cu_stg2["CU_loc_left"].item(), cu_stg2["CU_loc_top"].item())

                    # Split and compute positions
                    positions_stg3, sizes_stg3 = split(cu_size_stg2, cu_pos_stg2, cu_stg2["split"].item())

                    # Stage 3
                    for p2 in range(len(positions_stg3)):
                        # Get CU
                        cu_stg3 = find_cu(df_cus, CTU, positions_stg3[p2], sizes_stg3[p2])

                        if type(cu_stg3) != bool:  # If not found, it will be ignored

                            # Stop search
                            if cu_stg3["split"].item() == 0:
                                # Build dic until this point and add line to list
                                structed_cus.append(build_entry(CTU["all_info"], cu_stg2["all_info"].item(),
                                                                cu_stg3["all_info"].item()))

                            else:
                                # Split the current CU
                                cu_size_stg3 = (cu_stg3["width"].item(), cu_stg3["height"].item())
                                cu_pos_stg3 = (cu_stg3["CU_loc_left"].item(), cu_stg3["CU_loc_top"].item())

                                # Split and compute positions
                                positions_stg4, sizes_stg4 = split(cu_size_stg3, cu_pos_stg3, cu_stg3["split"].item())

                                # Stage 4
                                for p3 in range(len(positions_stg4)):
                                    # Get CU
                                    cu_stg4 = find_cu(df_cus, CTU, positions_stg4[p3], sizes_stg4[p3])

                                    if type(cu_stg4) != bool:  # If not found, it will be ignored

                                        # Stop search
                                        if cu_stg4["split"].item() == 0:
                                            # Build dic until this point and add line to list
                                            structed_cus.append(
                                                build_entry(CTU["all_info"], cu_stg2["all_info"].item(),
                                                            cu_stg3["all_info"].item(),
                                                            cu_stg4["all_info"].item()))

                                        else:
                                            # Split the current CU
                                            cu_size_stg4 = (cu_stg4["width"].item(), cu_stg4["height"].item())
                                            cu_pos_stg4 = (cu_stg4["CU_loc_left"].item(), cu_stg4["CU_loc_top"].item())

                                            # Split and compute positions
                                            positions_stg5, sizes_stg5 = split(cu_size_stg4, cu_pos_stg4,
                                                                               cu_stg4["split"].item())

                                            # Stage 5
                                            for p4 in range(len(positions_stg5)):
                                                # Get CU
                                                cu_stg5 = find_cu(df_cus, CTU, positions_stg5[p4], sizes_stg5[p4])

                                                if type(cu_stg5) != bool:  # If not found, it will be ignored

                                                    # Stop search
                                                    if cu_stg5["split"].item() == 0:
                                                        # Build dic until this point and add line to list
                                                        structed_cus.append(
                                                            build_entry(CTU["all_info"], cu_stg2["all_info"].item(),
                                                                        cu_stg3["all_info"].item(),
                                                                        cu_stg4["all_info"].item(),
                                                                        cu_stg5["all_info"].item()))

                                                    else:
                                                        # Split the current CU
                                                        cu_size_stg5 = (
                                                        cu_stg5["width"].item(), cu_stg5["height"].item())
                                                        cu_pos_stg5 = (
                                                        cu_stg5["CU_loc_left"].item(), cu_stg5["CU_loc_top"].item())

                                                        # Split and compute positions
                                                        positions_stg6, sizes_stg6 = split(cu_size_stg5, cu_pos_stg5,
                                                                                           cu_stg5["split"].item())

                                                        # Stage 6
                                                        for p5 in range(len(positions_stg6)):
                                                            # Get CU
                                                            cu_stg6 = find_cu(df_cus, CTU, positions_stg6[p5],
                                                                              sizes_stg6[p5])
                                                            if type(cu_stg6) != bool:
                                                                # Build dic until this point and add line to list
                                                                structed_cus.append(
                                                                    build_entry(CTU["all_info"],
                                                                                cu_stg2["all_info"].item(),
                                                                                cu_stg3["all_info"].item(),
                                                                                cu_stg4["all_info"].item(),
                                                                                cu_stg5["all_info"].item(),
                                                                                cu_stg6["all_info"].item()))

    return structed_cus

def split(size, pos, split_mode):
    """!
    @brief Split a CU in one of the specific modes (quad tree, binary vert tree, binary horz tree, threenary vert tree, etc)

    @param [in] size: Size of the CU (width, height)
    @param [in] pos: Position of the CU (width, height)
    @param [out] new_positions: Output of tuple with the positions of the CUs
    @param [out] new_sizes: Output of tuple with the sizes of the CUs
    """

    # Initialize variable
    CU_h = size[1]
    CU_w = size[0]
    left_pos = pos[0]
    top_pos = pos[1]
    new_positions = []
    new_sizes = []

    if (split_mode == 0):  # Non-split
        new_positions.append(pos)
        new_sizes.append(size)

    elif (split_mode == 1):  # Quad tree
        # Create new CUs
        position1 = pos
        position2 = (left_pos+int(CU_w / 2), top_pos)
        position3 = (left_pos, top_pos+int(CU_h / 2))
        position4 = (left_pos+int(CU_w / 2), top_pos+int(CU_h / 2))

        size1 = (int(CU_w/2), CU_h/2)
        size2 = (int(CU_w/2), CU_h/2)
        size3 = (int(CU_w/2), CU_h/2)
        size4 = (int(CU_w/2), CU_h/2)

        # Merge all
        new_positions = [position1, position2, position3, position4]
        new_sizes = [size1, size2, size3, size4]

    elif (split_mode == 3):  # Binary Vert tree
        # Create new CUs
        position1 = pos
        position2 = (left_pos + int(CU_w / 2), top_pos)

        size1 = (int(CU_w/2), CU_h)
        size2 = (int(CU_w/2), CU_h)

        # Merge all
        new_positions = [position1, position2]
        new_sizes = [size1, size2]

    elif (split_mode == 2):  # Binary Horz tree
        # Create new CUs
        position1 = pos
        position2 = (left_pos, top_pos + int(CU_h / 2))

        size1 = (CU_w, int(CU_h/2))
        size2 = (CU_w, int(CU_h/2))

        # Merge all
        new_positions = [position1, position2]
        new_sizes = [size1, size2]

    elif (split_mode == 5):  # Ternary Vert tree
        # Split in 3 parts with the ratio of 1,2,1
        # Create new CUs
        position1 = pos
        position2 = (left_pos + int(CU_w / 4), top_pos)
        position3 = (left_pos + int(3 * CU_w / 4), top_pos)

        size1 = (int(CU_w/4), CU_h)
        size2 = (int(CU_w/2), CU_h)
        size3 = (int(CU_w/4), CU_h)

        # Merge all
        new_positions = [position1, position2, position3]
        new_sizes = [size1, size2, size3]

    elif (split_mode == 4):  # Ternary Horz tree
        # Split in 3 parts with the ratio of 1,2,1
        # Create new CUs
        position1 = pos
        position2 = (left_pos, top_pos + int(CU_h / 4))
        position3 = (left_pos, top_pos + int(3 * CU_h / 4))

        size1 = (CU_w, int(CU_h/4))
        size2 = (CU_w, int(CU_h/2))
        size3 = (CU_w, int(CU_h/4))

        # Merge all
        new_positions = [position1, position2, position3]
        new_sizes = [size1, size2, size3]

    else:
        raise Exception("Wrong split number!!")

    return new_positions, new_sizes

def transform_raw_dataset(dic):
    """!
    @brief Transform raw dataset (dictionary with information of all datasets) and convert it to a list of dictionaries
    * List entry: pic_name | color_ch | POC | CU_loc_left | ... | split
    CU oriented style

    @param [in] dic: Dictionary containing all the raw data
    @param [out] lst_dicts: List of dictionaries (entries of the information of each CU)
    """

    # Initialize list
    lst_dicts = []

    # Add best split to the dictionary
    dic_new = add_best_split(dic)

    for key in dic_new.keys():
        # Get picture name
        pic_name = key.split("_")
        pic_name = pic_name[0] + "_" + pic_name[1] + "_" + pic_name[2]

        for key2 in dic_new[key].keys():
            mini_dict = {"pic_name": pic_name, "color_ch": dic_new[key][key2]["color_ch"], "POC": dic_new[key][key2]["POC"],
                         "CU_loc_left": dic_new[key][key2]["CU_loc_left"], "CU_loc_top": dic_new[key][key2]["CU_loc_top"],
                         "CU_w": dic_new[key][key2]["CU_w"], "CU_h": dic_new[key][key2]["CU_h"],
                         "RD_mode0": dic_new[key][key2]["RD_mode0"], "RD_mode1": dic_new[key][key2]["RD_mode1"],
                         "RD_mode2": dic_new[key][key2]["RD_mode2"], "RD_mode3": dic_new[key][key2]["RD_mode3"],
                         "RD_mode4": dic_new[key][key2]["RD_mode4"],
                         "RD_mode5": dic_new[key][key2]["RD_mode5"], "split": dic_new[key][key2]["split"]}

            lst_dicts.append(mini_dict)

    return lst_dicts

def get_files_from_folder(path, endswith=".yuv"):
    """!
    @brief This function obtains the name of all .yuv files in a given path

    @param [in] path: Path containing the files
    @param [out] files_list: List containing all the names of the .yuv and .hif files
    """

    # Initialize list
    files_lst = []

    # Add file name to list, if the file ends with '.yuv'
    for x in os.listdir(path):

        if x.endswith(endswith):
            files_lst.append(x)

    return files_lst


def get_num_frames(path, name, width, height):
    """!
    @brief Get number of frames in yuv file

    @param [in] path: Path containing dataset
    @param [in] name: Name of the file where the file is located
    @param [in] width: Width of the picture
    @param [in] height: Height of the picture
    @param [out] num_frames: Number of frames that the file contain
    """

    # Image location
    img_path = path + "/" + name
    print("Image path: ", img_path)
    vid_YUV = VideoCaptureYUV(img_path, (height, width))

    num_frames = 0
    # Read until video is completed
    while (True):

        # Capture frame-by-frame
        ret, frame = vid_YUV.read()
        if ret == True:
            num_frames += 1

        # Break the loop
        else:
            break

    return num_frames


def get_file_metadata_info(path, name):
    """!
    @brief Retrieves information about the YUV file info (framerate, width and height and number of frames)

    @param [in] path: Path containing dataset
    @param [in] name: Name of the file where the file is located
    @param [out] file_info: Dictionary with information about the yuv file (dimensions, frame rate and number of frames) or a boolean value indicating that there is no informations
    """

    # Initialize variable
    file_info = {}

    ## Look for size
    # Look for the type "Number x Number"
    size = re.findall("\d+x\d+", name)
    if len(size) == 1:
        # Get size
        size = list(map(int, re.findall('\d+', size[0])))  # Obtain the values in integer
        file_info["width"] = size[0]
        file_info["height"] = size[1]

        # Look for fps
        framerate = re.findall("_\d\d_|_\d\d\.", name)
        if len(framerate) == 1:
            file_info["frame_rate"] = int(framerate[0][1:3])

        else:
            file_info["frame_rate"] = 30  # Default frame_rate

    # Look for the type cif
    size = re.findall("_cif", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 352
        file_info["height"] = 288
        # Frame rate
        file_info["frame_rate"] = 30

    # Look for the type sif
    size = re.findall("_sif", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 352
        file_info["height"] = 240
        # Frame rate
        file_info["frame_rate"] = 30

    # Look for the type 4cif
    size = re.findall("_4cif", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 704
        file_info["height"] = 576
        # Frame rate
        file_info["frame_rate"] = 30

    # Look for the type 1080p
    size = re.findall("1080p\d*", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 1920
        file_info["height"] = 1080
        # Frame rate
        framerate = list(map(int, re.findall('\d+', size[0])))  # Get all numbers from string
        if len(framerate) == 2:
            file_info["frame_rate"] = framerate[1]  # Get frame rate

        else:
            file_info["frame_rate"] = 30  # Default frame rate

    # Look for the type 720p
    size = re.findall("720p\d*", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 1280
        file_info["height"] = 720
        # Frame rate
        framerate = list(map(int, re.findall('\d+', size[0])))  # Get all numbers from string
        if len(framerate) == 2:
            file_info["frame_rate"] = framerate[1]  # Get frame rate

        else:
            file_info["frame_rate"] = 30  # Default frame rate

    # Get number of frames
    file_info["num_frames"] = get_num_frames(path, name, file_info["width"], file_info["height"])

    if len(file_info) == 0:
        return False

    else:
        return file_info

def get_file_metadata_info_mod(name):
    """!
    @brief Retrieves information about the YUV file info (framerate, width and height ). This version doesn't compute the number of frames.

    @param [in] name: Name of the file where the file is located
    @param [out] file_info: Dictionary with information about the yuv file (dimensions and frame rate) or a boolean value indicating that there is no informations
    """

    # Initialize variable
    file_info = {}

    ## Look for size
    # Look for the type "Number x Number"
    size = re.findall("\d+x\d+", name)
    if len(size) == 1:
        # Get size
        size = list(map(int, re.findall('\d+', size[0])))  # Obtain the values in integer
        file_info["width"] = size[0]
        file_info["height"] = size[1]

        # Look for fps
        framerate = re.findall("_\d\d_|_\d\d\.", name)
        if len(framerate) == 1:
            file_info["frame_rate"] = int(framerate[0][1:3])

        else:
            file_info["frame_rate"] = 30  # Default frame_rate

    # Look for the type cif
    size = re.findall("_cif", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 352
        file_info["height"] = 288
        # Frame rate
        file_info["frame_rate"] = 30

    # Look for the type sif
    size = re.findall("_sif", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 352
        file_info["height"] = 240
        # Frame rate
        file_info["frame_rate"] = 30

    # Look for the type 4cif
    size = re.findall("_4cif", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 704
        file_info["height"] = 576
        # Frame rate
        file_info["frame_rate"] = 30

    # Look for the type 1080p
    size = re.findall("1080p\d*", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 1920
        file_info["height"] = 1080
        # Frame rate
        framerate = list(map(int, re.findall('\d+', size[0])))  # Get all numbers from string
        if len(framerate) == 2:
            file_info["frame_rate"] = framerate[1]  # Get frame rate

        else:
            file_info["frame_rate"] = 30  # Default frame rate

    # Look for the type 720p
    size = re.findall("720p\d*", name)
    if len(size) == 1:
        # Size
        file_info["width"] = 1280
        file_info["height"] = 720
        # Frame rate
        framerate = list(map(int, re.findall('\d+', size[0])))  # Get all numbers from string
        if len(framerate) == 2:
            file_info["frame_rate"] = framerate[1]  # Get frame rate

        else:
            file_info["frame_rate"] = 30  # Default frame rate

    if len(file_info) == 0:
        return False

    else:
        return file_info

def encode_dataset(d_path="C:\\Users\\Raul\\Dropbox\\Dataset",
                   e_path="C:\\Users\\Raul\\Documents\\GitHub\\CPIV\\VTM-7.0_Data\\bin\\vs16\\msvc-19.24\\x86_64\\release",
                   ts=1, QP=32):
    """!
    @brief This function encodes the entire dataset with in a given path

    @param [in] d_path: Path containing the dataset with the files to encode (this path can not contain spaces)
    @param [in] e_path: Path containing the encoder and configurations for it
    @param [in] ts: Temporal Subsample Ratio (ts é o parametro que controla a quantidade de frames)
    @param [in] QP: Quantization parameter

    """

    # Obtain name of the files that were already encoded
    files_encoded = get_files_from_folder(e_path, endswith='.dat')

    # Obtain the name of the files to encode
    files_list = get_files_from_folder(d_path)

    # Remove from the files to encode list, the files that were already encoded
    for file_enc in files_encoded:
        for file in files_list:
            if file_enc[:-8] == file[:-4]:
                files_list.remove(file)

    # Obtain info of each file and create dict
    dict_files_n_info = {}
    for file in files_list:
        file_info = get_file_metadata_info(d_path, file)  # Obtain info specified in the file name
        if not isinstance(file_info, bool):  # Confirm if the data is valid
            dict_files_n_info[file] = file_info

    # Verify which OS is being used
    encoder_app = ''
    from sys import platform
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        encoder_app = "./EncoderAppStatic"
    elif platform == "win32":
        encoder_app = "EncoderApp"
    else:
        raise Exception("Operating system not identified")

    # Encode each file
    os.chdir(e_path)  # Move current process to the Encoder directory
    for f_name in dict_files_n_info.keys():
        # Encode file
        cmd = encoder_app + " -c " + "encoder_intra_vtm.cfg -i " + d_path + "/" + f_name + " -wdt " + str(
            dict_files_n_info[f_name]["width"]) + " -hgt " + str(dict_files_n_info[f_name]["height"]) + " -fr " + str(
            dict_files_n_info[f_name]["frame_rate"]) + " --FramesToBeEncoded=" + str(
            dict_files_n_info[f_name]["num_frames"]) + " -ts " + str(ts) + " -q " + str(QP)

        print("Running command:", cmd)
        print("Current dir: ", os.getcwd())
        os.system(cmd)

        # Wait: So that the CUInfoCost.dat is created
        time.sleep(10)

        # Save CUInfoCost.dat with a diferent name
        new_name = f_name[:-4] + "_Enc.dat"
        os.rename("CUInfoCost.dat", new_name)

def compute_split_per_depth(d_path):
    """!
    @brief Compute the percentage and number of splits per depth of the partitiooning scheme

    @param [in] d_path: Path with the files containing with the cus sequences
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """
    
    print("Inicializing..")
    # Initialize variables
    pm = {0: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            1: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            2: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            3: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            4: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}, 
            5: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}}
    am = {0: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          1: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          2: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          3: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          4: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}, 
          5: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}}

    # Pandas structure to keep track of the CUs
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]
    pd_full = pd.DataFrame(columns=columns)

    # Obtain the name of the files
    files_list = get_files_from_folder(d_path, ".txt")

    print("Computing..")

    # Change dir workspace
    os.chdir(d_path)

    for path in files_list:
        # Process data
        lst_dicts = file2lst(path[:-4])

        # Count
        for dic in lst_dicts:

            stg_1 = dic["stg_1"]
            stg_2 = dic["stg_2"]
            stg_3 = dic["stg_3"]
            stg_4 = dic["stg_4"]
            stg_5 = dic["stg_5"]
            stg_6 = dic["stg_6"]

            if type(stg_1) != list:
                if len(pd_full[(pd_full["POC"] ==  dic["POC"]) & (pd_full["pic_name"] == dic["pic_name"]) & (pd_full["cu_size_w"] == stg_1["cu_size"]["width"]) & \
                    (pd_full["cu_size_h"] == stg_1["cu_size"]["height"]) & \
                    (pd_full["cu_pos_y"] == stg_1["cu_pos"]["CU_loc_top"]) &\
                    (pd_full["cu_pos_x"] == stg_1["cu_pos"]["CU_loc_left"])]) == 0 :
                        am[0][stg_1["split"]] += 1
                        pd_row = pd.DataFrame({"POC": [stg_1["POC"]], "pic_name": stg_1["pic_name"], "cu_pos_x": stg_1["cu_pos"]["CU_loc_left"],\
                            "cu_pos_y": stg_1["cu_pos"]["CU_loc_top"], "cu_size_h": stg_1["cu_size"]["height"],\
                            "cu_size_w": stg_1["cu_size"]["width"], "split": stg_1["split"]})
                        pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

            if type(stg_2) != list:
                if len(pd_full[(pd_full["POC"] ==  dic["POC"]) & (pd_full["pic_name"] == dic["pic_name"]) & (pd_full["cu_size_w"] == stg_2["cu_size"]["width"]) & \
                    (pd_full["cu_size_h"] == stg_2["cu_size"]["height"]) & \
                    (pd_full["cu_pos_y"] == stg_2["cu_pos"]["CU_loc_top"]) &\
                    (pd_full["cu_pos_x"] == stg_2["cu_pos"]["CU_loc_left"])]) == 0:
                        am[1][stg_2["split"]] += 1
                        pd_row = pd.DataFrame({"POC": [stg_2["POC"]], "pic_name": stg_2["pic_name"], "cu_pos_x": stg_2["cu_pos"]["CU_loc_left"],\
                            "cu_pos_y": stg_2["cu_pos"]["CU_loc_top"], "cu_size_h": stg_2["cu_size"]["height"],\
                            "cu_size_w": stg_2["cu_size"]["width"], "split": stg_2["split"]})
                        pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

            if type(stg_3) != list:
                if len(pd_full[(pd_full["POC"] ==  dic["POC"]) & (pd_full["pic_name"] == dic["pic_name"]) & (pd_full["cu_size_w"] == stg_3["cu_size"]["width"]) & \
                    (pd_full["cu_size_h"] == stg_3["cu_size"]["height"]) & \
                    (pd_full["cu_pos_y"] == stg_3["cu_pos"]["CU_loc_top"]) &\
                    (pd_full["cu_pos_x"] == stg_3["cu_pos"]["CU_loc_left"])]) == 0:
                        am[2][stg_3["split"]] += 1
                        pd_row = pd.DataFrame({"POC": [stg_3["POC"]], "pic_name": stg_3["pic_name"], "cu_pos_x": stg_3["cu_pos"]["CU_loc_left"],\
                            "cu_pos_y": stg_3["cu_pos"]["CU_loc_top"], "cu_size_h": stg_3["cu_size"]["height"],\
                            "cu_size_w": stg_3["cu_size"]["width"], "split": stg_3["split"]})
                        pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

            if type(stg_4) != list:
                if len(pd_full[(pd_full["POC"] ==  dic["POC"]) & (pd_full["pic_name"] == dic["pic_name"]) & (pd_full["cu_size_w"] == stg_4["cu_size"]["width"]) & \
                    (pd_full["cu_size_h"] == stg_4["cu_size"]["height"]) & \
                    (pd_full["cu_pos_y"] == stg_4["cu_pos"]["CU_loc_top"]) &\
                    (pd_full["cu_pos_x"] == stg_4["cu_pos"]["CU_loc_left"])]) == 0:
                        am[3][stg_4["split"]] += 1
                        pd_row = pd.DataFrame({"POC": [stg_4["POC"]], "pic_name": stg_4["pic_name"], "cu_pos_x": stg_4["cu_pos"]["CU_loc_left"],\
                            "cu_pos_y": stg_4["cu_pos"]["CU_loc_top"], "cu_size_h": stg_4["cu_size"]["height"],\
                            "cu_size_w": stg_4["cu_size"]["width"], "split": stg_4["split"]})
                        pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

            if type(stg_5) != list:
                if len(pd_full[(pd_full["POC"] ==  dic["POC"]) & (pd_full["pic_name"] == dic["pic_name"]) & (pd_full["cu_size_w"] == stg_5["cu_size"]["width"]) & \
                    (pd_full["cu_size_h"] == stg_5["cu_size"]["height"]) & \
                    (pd_full["cu_pos_y"] == stg_5["cu_pos"]["CU_loc_top"]) &\
                    (pd_full["cu_pos_x"] == stg_5["cu_pos"]["CU_loc_left"])]) == 0:
                        am[4][stg_5["split"]] += 1
                        pd_row = pd.DataFrame({"POC": [stg_5["POC"]], "pic_name": stg_5["pic_name"], "cu_pos_x": stg_5["cu_pos"]["CU_loc_left"],\
                            "cu_pos_y": stg_5["cu_pos"]["CU_loc_top"], "cu_size_h": stg_5["cu_size"]["height"],\
                            "cu_size_w": stg_5["cu_size"]["width"], "split": stg_5["split"]})
                        pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)

            if type(stg_6) != list:
                if len(pd_full[(pd_full["POC"] ==  dic["POC"]) & (pd_full["pic_name"] == dic["pic_name"]) & (pd_full["cu_size_w"] == stg_6["cu_size"]["width"]) & \
                    (pd_full["cu_size_h"] == stg_6["cu_size"]["height"]) & \
                    (pd_full["cu_pos_y"] == stg_6["cu_pos"]["CU_loc_top"]) &\
                    (pd_full["cu_pos_x"] == stg_6["cu_pos"]["CU_loc_left"])]) == 0:
                        am[5][stg_6["split"]] += 1
                    
                        pd_row = pd.DataFrame({"POC": [stg_6["POC"]], "pic_name": stg_6["pic_name"], "cu_pos_x": stg_6["cu_pos"]["CU_loc_left"],\
                            "cu_pos_y": stg_6["cu_pos"]["CU_loc_top"], "cu_size_h": stg_6["cu_size"]["height"],\
                            "cu_size_w": stg_6["cu_size"]["width"], "split": stg_6["split"]})
                        pd_full = pd.concat([pd_full, pd_row], ignore_index=True, axis=0)


    print("Preparing results..")
    print()
    # Find out the total amount of partitions per depth
    total = {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for key in total.keys():
        for key2 in am[0].keys():
            total[key] += am[key][key2]

    # Compute proportions per depth
    for key in am.keys():
        for key2 in pm[0].keys():
            pm[key][key2] = am[key][key2] / total[key]

    print("Results:")
    print()
    print("Amount of partitions per depth:", am)
    print()
    print("Proportion of partitions per depth:", pm)

    return pm, am


def compute_split_per_depth_v2(d_path):
    """!
    @brief Compute the percentage and number of splits per depth of the partitiooning scheme. This version uses just dataframe

    @param [in] d_path: Path with the files containing with the cus sequences
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """
    
    print("Inicializing..")
    pm = {0: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            1: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            2: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            3: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            4: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}, 
            5: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}}
    am = {0: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          1: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          2: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          3: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          4: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}, 
          5: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}}


    # Obtain the name of the files
    files_list = get_files_from_folder(d_path, ".txt")

    print("Computing..")

    # Change dir workspace
    os.chdir(d_path)

    for k in range(len(files_list)):
        # Process data
        lst_dict = file2lst(files_list[k][:-4])

        # Add to data frame
        for k2 in range(len(lst_dict)):
            
            try:
                # Modify structure
                if type(lst_dict[k2]["stg_1"]) != list:
                    lst_dict[k2]["stg_1"]["x"] = lst_dict[k2]["stg_1"]["cu_pos"]["CU_loc_left"]
                    lst_dict[k2]["stg_1"]["y"] = lst_dict[k2]["stg_1"]["cu_pos"]["CU_loc_top"]
                    lst_dict[k2]["stg_1"]["h"] = lst_dict[k2]["stg_1"]["cu_size"]["height"]
                    lst_dict[k2]["stg_1"]["w"] = lst_dict[k2]["stg_1"]["cu_size"]["width"]
                    del lst_dict[k2]["stg_1"]["cu_size"]
                    del lst_dict[k2]["stg_1"]["cu_pos"]
            except:
                pass

            try:
                if type(lst_dict[k2]["stg_2"]) != list:
                    lst_dict[k2]["stg_2"]["x"] = lst_dict[k2]["stg_2"]["cu_pos"]["CU_loc_left"]
                    lst_dict[k2]["stg_2"]["y"] = lst_dict[k2]["stg_2"]["cu_pos"]["CU_loc_top"]
                    lst_dict[k2]["stg_2"]["h"] = lst_dict[k2]["stg_2"]["cu_size"]["height"]
                    lst_dict[k2]["stg_2"]["w"] = lst_dict[k2]["stg_2"]["cu_size"]["width"]
                    del lst_dict[k2]["stg_2"]["cu_size"]
                    del lst_dict[k2]["stg_2"]["cu_pos"]
            except:
                pass
            
            try:
                if type(lst_dict[k2]["stg_3"]) != list:
                    lst_dict[k2]["stg_3"]["x"] = lst_dict[k2]["stg_3"]["cu_pos"]["CU_loc_left"]
                    lst_dict[k2]["stg_3"]["y"] = lst_dict[k2]["stg_3"]["cu_pos"]["CU_loc_top"]
                    lst_dict[k2]["stg_3"]["h"] = lst_dict[k2]["stg_3"]["cu_size"]["height"]
                    lst_dict[k2]["stg_3"]["w"] = lst_dict[k2]["stg_3"]["cu_size"]["width"]
                    del lst_dict[k2]["stg_3"]["cu_size"]
                    del lst_dict[k2]["stg_3"]["cu_pos"]
            except:
                pass
            
            try:
                if type(lst_dict[k2]["stg_4"]) != list:
                    lst_dict[k2]["stg_4"]["x"] = lst_dict[k2]["stg_4"]["cu_pos"]["CU_loc_left"]
                    lst_dict[k2]["stg_4"]["y"] = lst_dict[k2]["stg_4"]["cu_pos"]["CU_loc_top"]
                    lst_dict[k2]["stg_4"]["h"] = lst_dict[k2]["stg_4"]["cu_size"]["height"]
                    lst_dict[k2]["stg_4"]["w"] = lst_dict[k2]["stg_4"]["cu_size"]["width"]
                    del lst_dict[k2]["stg_4"]["cu_size"]
                    del lst_dict[k2]["stg_4"]["cu_pos"]
            except:
                pass

            try:
                if type(lst_dict[k2]["stg_5"]) != list:
                    lst_dict[k2]["stg_5"]["x"] = lst_dict[k2]["stg_5"]["cu_pos"]["CU_loc_left"]
                    lst_dict[k2]["stg_5"]["y"] = lst_dict[k2]["stg_5"]["cu_pos"]["CU_loc_top"]
                    lst_dict[k2]["stg_5"]["h"] = lst_dict[k2]["stg_5"]["cu_size"]["height"]
                    lst_dict[k2]["stg_5"]["w"] = lst_dict[k2]["stg_5"]["cu_size"]["width"]
                    del lst_dict[k2]["stg_5"]["cu_size"]
                    del lst_dict[k2]["stg_5"]["cu_pos"]
            except:
                pass

            try:    
                if type(lst_dict[k2]["stg_6"]) != list:
                    lst_dict[k2]["stg_6"]["x"] = lst_dict[k2]["stg_6"]["cu_pos"]["CU_loc_left"]
                    lst_dict[k2]["stg_6"]["y"] = lst_dict[k2]["stg_6"]["cu_pos"]["CU_loc_top"]
                    lst_dict[k2]["stg_6"]["h"] = lst_dict[k2]["stg_6"]["cu_size"]["height"]
                    lst_dict[k2]["stg_6"]["w"] = lst_dict[k2]["stg_6"]["cu_size"]["width"]
                    del lst_dict[k2]["stg_6"]["cu_size"]
                    del lst_dict[k2]["stg_6"]["cu_pos"]
            except:
                pass

            if k2 == 0:
                df_data_stg1 = pd.DataFrame(lst_dict[k2]["stg_1"], index=[0])
                df_data_stg2 = pd.DataFrame(lst_dict[k2]["stg_2"], index=[0])
                df_data_stg3 = pd.DataFrame(lst_dict[k2]["stg_3"], index=[0])
                df_data_stg4 = pd.DataFrame(lst_dict[k2]["stg_4"], index=[0])
                df_data_stg5 = pd.DataFrame(lst_dict[k2]["stg_5"], index=[0])
                df_data_stg6 = pd.DataFrame(lst_dict[k2]["stg_6"], index=[0])
            
            else: 
                df_data_stg1 = pd.concat([df_data_stg1, pd.DataFrame(lst_dict[k2]["stg_1"], index=[0])], ignore_index=True, axis=0)
                df_data_stg2 = pd.concat([df_data_stg2, pd.DataFrame(lst_dict[k2]["stg_2"], index=[0])], ignore_index=True, axis=0)
                df_data_stg3 = pd.concat([df_data_stg3, pd.DataFrame(lst_dict[k2]["stg_3"], index=[0])], ignore_index=True, axis=0)
                df_data_stg4 = pd.concat([df_data_stg4, pd.DataFrame(lst_dict[k2]["stg_4"], index=[0])], ignore_index=True, axis=0)
                df_data_stg5 = pd.concat([df_data_stg5, pd.DataFrame(lst_dict[k2]["stg_5"], index=[0])], ignore_index=True, axis=0)
                df_data_stg6 = pd.concat([df_data_stg6, pd.DataFrame(lst_dict[k2]["stg_6"], index=[0])], ignore_index=True, axis=0)

            # Drop duplicates
            df_data_stg1.drop_duplicates()
            df_data_stg2.drop_duplicates()
            df_data_stg3.drop_duplicates()
            df_data_stg4.drop_duplicates()
            df_data_stg5.drop_duplicates()
            df_data_stg6.drop_duplicates()

        # Stage 2
        am[0][0] += len(df_data_stg1.loc[(df_data_stg1["split"] == 0)])
        am[0][1] += len(df_data_stg1.loc[(df_data_stg1["split"] == 1)])
        am[0][2] += len(df_data_stg1.loc[(df_data_stg1["split"] == 2)])
        am[0][3] += len(df_data_stg1.loc[(df_data_stg1["split"] == 3)])
        am[0][4] += len(df_data_stg1.loc[(df_data_stg1["split"] == 4)])
        am[0][5] += len(df_data_stg1.loc[(df_data_stg1["split"] == 5)])
        
        # Stage 2
        am[1][0] += len(df_data_stg2.loc[(df_data_stg2["split"] == 0)])
        am[1][1] += len(df_data_stg2.loc[(df_data_stg2["split"] == 1)])
        am[1][2] += len(df_data_stg2.loc[(df_data_stg2["split"] == 2)])
        am[1][3] += len(df_data_stg2.loc[(df_data_stg2["split"] == 3)])
        am[1][4] += len(df_data_stg2.loc[(df_data_stg2["split"] == 4)])
        am[1][5] += len(df_data_stg2.loc[(df_data_stg2["split"] == 5)])

        # Stage 3
        am[2][0] += len(df_data_stg3.loc[(df_data_stg3["split"] == 0)])
        am[2][1] += len(df_data_stg3.loc[(df_data_stg3["split"] == 1)])
        am[2][2] += len(df_data_stg3.loc[(df_data_stg3["split"] == 2)])
        am[2][3] += len(df_data_stg3.loc[(df_data_stg3["split"] == 3)])
        am[2][4] += len(df_data_stg3.loc[(df_data_stg3["split"] == 4)])
        am[2][5] += len(df_data_stg3.loc[(df_data_stg3["split"] == 5)])

        # Stage 4
        am[3][0] += len(df_data_stg4.loc[(df_data_stg4["split"] == 0)])
        am[3][1] += len(df_data_stg4.loc[(df_data_stg4["split"] == 1)])
        am[3][2] += len(df_data_stg4.loc[(df_data_stg4["split"] == 2)])
        am[3][3] += len(df_data_stg4.loc[(df_data_stg4["split"] == 3)])
        am[3][4] += len(df_data_stg4.loc[(df_data_stg4["split"] == 4)])
        am[3][5] += len(df_data_stg4.loc[(df_data_stg4["split"] == 5)])

        # Stage 5
        am[4][0] += len(df_data_stg5.loc[(df_data_stg5["split"] == 0)])
        am[4][1] += len(df_data_stg5.loc[(df_data_stg5["split"] == 1)])
        am[4][2] += len(df_data_stg5.loc[(df_data_stg5["split"] == 2)])
        am[4][3] += len(df_data_stg5.loc[(df_data_stg5["split"] == 3)])
        am[4][4] += len(df_data_stg5.loc[(df_data_stg5["split"] == 4)])
        am[4][5] += len(df_data_stg5.loc[(df_data_stg5["split"] == 5)])

        # Stage 6
        am[5][0] += len(df_data_stg6.loc[(df_data_stg6["split"] == 0)])
        am[5][1] += len(df_data_stg6.loc[(df_data_stg6["split"] == 1)])
        am[5][2] += len(df_data_stg6.loc[(df_data_stg6["split"] == 2)])
        am[5][3] += len(df_data_stg6.loc[(df_data_stg6["split"] == 3)])
        am[5][4] += len(df_data_stg6.loc[(df_data_stg6["split"] == 4)])
        am[5][5] += len(df_data_stg6.loc[(df_data_stg6["split"] == 5)])

        

    print("Preparing results..")
    print()
    # Find out the total amount of partitions per depth
    total = {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for key in total.keys():
        for key2 in am[0].keys():
            total[key] += am[key][key2]

    # Compute proportions per depth
    for key in am.keys():
        for key2 in pm[0].keys():
            pm[key][key2] = am[key][key2] / total[key]

    print("Results:")
    print()
    print("Amount of partitions per depth:", am)
    print()
    print("Proportion of partitions per depth:", pm)

    return pm, am


def compute_split_per_depth_v3(d_path):
    """!
    @brief Compute the percentage and number of splits per depth of the partitiooning scheme. This version uses just list comprehension

    @param [in] d_path: Path with the files containing with the cus sequences
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """
    
    print("Inicializing..")
    pm = {0: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            1: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            2: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            3: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0},
            4: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}, 
            5: {0: 0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}}
    am = {0: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          1: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          2: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          3: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0},
          4: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}, 
          5: {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}}


    # Obtain the name of the files
    files_list = get_files_from_folder(d_path, ".txt")

    print("Computing..")

    # Change dir workspace
    os.chdir(d_path)

    for k in range(len(files_list)):
        # Process data
        lst_dict = file2lst(files_list[k][:-4])
        
        # Get all stages
        stg1_lsts = [d['stg_1'] for d in lst_dict if type(d['stg_1']) != list]
        stg2_lsts = [d['stg_2'] for d in lst_dict if type(d['stg_2']) != list]
        stg3_lsts = [d['stg_3'] for d in lst_dict if type(d['stg_3']) != list]
        stg4_lsts = [d['stg_4'] for d in lst_dict if type(d['stg_4']) != list]
        stg5_lsts = [d['stg_5'] for d in lst_dict if type(d['stg_5']) != list]
        stg6_lsts = [d['stg_6'] for d in lst_dict if type(d['stg_6']) != list]

        # Add cu_size and cu_width to list of dictionaries
        #stg1_size = [{ for (d1, d2) in zip(dicty, dicty["cu_size"]) if "cu_" in k or k is k} {{**dicty[k], **dicty[k]["cu_size"]}} for dicty in stg1_lsts]
        stg1_lsts = [{**dicty, **dicty["cu_size"]} for dicty in stg1_lsts]
        stg1_lsts = [{**dicty, **dicty["cu_pos"]} for dicty in stg1_lsts]
        stg1_lsts = [{key : val for key, val in sub.items() if key != "cu_pos"} for sub in stg1_lsts]
        stg1_lsts = [{key : val for key, val in sub.items() if key != "cu_size"} for sub in stg1_lsts]

        stg2_lsts = [{**dicty, **dicty["cu_size"]} for dicty in stg2_lsts]
        stg2_lsts = [{**dicty, **dicty["cu_pos"]} for dicty in stg2_lsts]
        stg2_lsts = [{key : val for key, val in sub.items() if key != "cu_pos"} for sub in stg2_lsts]
        stg2_lsts = [{key : val for key, val in sub.items() if key != "cu_size"} for sub in stg2_lsts]

        stg3_lsts = [{**dicty, **dicty["cu_size"]} for dicty in stg3_lsts]
        stg3_lsts = [{**dicty, **dicty["cu_pos"]} for dicty in stg3_lsts]
        stg3_lsts = [{key : val for key, val in sub.items() if key != "cu_pos"} for sub in stg3_lsts]
        stg3_lsts = [{key : val for key, val in sub.items() if key != "cu_size"} for sub in stg3_lsts]

        stg4_lsts = [{**dicty, **dicty["cu_size"]} for dicty in stg4_lsts]
        stg4_lsts = [{**dicty, **dicty["cu_pos"]} for dicty in stg4_lsts]
        stg4_lsts = [{key : val for key, val in sub.items() if key != "cu_pos"} for sub in stg4_lsts]
        stg4_lsts = [{key : val for key, val in sub.items() if key != "cu_size"} for sub in stg4_lsts]

        stg5_lsts = [{**dicty, **dicty["cu_size"]} for dicty in stg5_lsts]
        stg5_lsts = [{**dicty, **dicty["cu_pos"]} for dicty in stg5_lsts]
        stg5_lsts = [{key : val for key, val in sub.items() if key != "cu_pos"} for sub in stg5_lsts]
        stg5_lsts = [{key : val for key, val in sub.items() if key != "cu_size"} for sub in stg5_lsts]

        stg6_lsts = [{**dicty, **dicty["cu_size"]} for dicty in stg6_lsts]
        stg6_lsts = [{**dicty, **dicty["cu_pos"]} for dicty in stg6_lsts]
        stg6_lsts = [{key : val for key, val in sub.items() if key != "cu_pos"} for sub in stg6_lsts]
        stg6_lsts = [{key : val for key, val in sub.items() if key != "cu_size"} for sub in stg6_lsts]

        # Remove duplicates
        stg1_lsts = [dict(t) for t in {tuple(d.items()) for d in stg1_lsts}]
        stg2_lsts = [dict(t) for t in {tuple(d.items()) for d in stg2_lsts}]
        stg3_lsts = [dict(t) for t in {tuple(d.items()) for d in stg3_lsts}]
        stg4_lsts = [dict(t) for t in {tuple(d.items()) for d in stg4_lsts}]
        stg5_lsts = [dict(t) for t in {tuple(d.items()) for d in stg5_lsts}]
        stg6_lsts = [dict(t) for t in {tuple(d.items()) for d in stg6_lsts}]

        # Create DataFrames
        df_data_stg1 = pd.DataFrame(stg1_lsts)
        df_data_stg2 = pd.DataFrame(stg2_lsts)
        df_data_stg3 = pd.DataFrame(stg3_lsts)
        df_data_stg4 = pd.DataFrame(stg4_lsts)
        df_data_stg5 = pd.DataFrame(stg5_lsts)
        df_data_stg6 = pd.DataFrame(stg6_lsts)


        # Add information to storage variable
        # Stage 1
        am[0][0] += len(df_data_stg1.loc[(df_data_stg1["split"] == 0)])
        am[0][1] += len(df_data_stg1.loc[(df_data_stg1["split"] == 1)])
        am[0][2] += len(df_data_stg1.loc[(df_data_stg1["split"] == 2)])
        am[0][3] += len(df_data_stg1.loc[(df_data_stg1["split"] == 3)])
        am[0][4] += len(df_data_stg1.loc[(df_data_stg1["split"] == 4)])
        am[0][5] += len(df_data_stg1.loc[(df_data_stg1["split"] == 5)])
        
        # Stage 2
        am[1][0] += len(df_data_stg2.loc[(df_data_stg2["split"] == 0)])
        am[1][1] += len(df_data_stg2.loc[(df_data_stg2["split"] == 1)])
        am[1][2] += len(df_data_stg2.loc[(df_data_stg2["split"] == 2)])
        am[1][3] += len(df_data_stg2.loc[(df_data_stg2["split"] == 3)])
        am[1][4] += len(df_data_stg2.loc[(df_data_stg2["split"] == 4)])
        am[1][5] += len(df_data_stg2.loc[(df_data_stg2["split"] == 5)])

        # Stage 3
        am[2][0] += len(df_data_stg3.loc[(df_data_stg3["split"] == 0)])
        am[2][1] += len(df_data_stg3.loc[(df_data_stg3["split"] == 1)])
        am[2][2] += len(df_data_stg3.loc[(df_data_stg3["split"] == 2)])
        am[2][3] += len(df_data_stg3.loc[(df_data_stg3["split"] == 3)])
        am[2][4] += len(df_data_stg3.loc[(df_data_stg3["split"] == 4)])
        am[2][5] += len(df_data_stg3.loc[(df_data_stg3["split"] == 5)])

        # Stage 4
        am[3][0] += len(df_data_stg4.loc[(df_data_stg4["split"] == 0)])
        am[3][1] += len(df_data_stg4.loc[(df_data_stg4["split"] == 1)])
        am[3][2] += len(df_data_stg4.loc[(df_data_stg4["split"] == 2)])
        am[3][3] += len(df_data_stg4.loc[(df_data_stg4["split"] == 3)])
        am[3][4] += len(df_data_stg4.loc[(df_data_stg4["split"] == 4)])
        am[3][5] += len(df_data_stg4.loc[(df_data_stg4["split"] == 5)])

        # Stage 5
        am[4][0] += len(df_data_stg5.loc[(df_data_stg5["split"] == 0)])
        am[4][1] += len(df_data_stg5.loc[(df_data_stg5["split"] == 1)])
        am[4][2] += len(df_data_stg5.loc[(df_data_stg5["split"] == 2)])
        am[4][3] += len(df_data_stg5.loc[(df_data_stg5["split"] == 3)])
        am[4][4] += len(df_data_stg5.loc[(df_data_stg5["split"] == 4)])
        am[4][5] += len(df_data_stg5.loc[(df_data_stg5["split"] == 5)])

        # Stage 6
        am[5][0] += len(df_data_stg6.loc[(df_data_stg6["split"] == 0)])
        am[5][1] += len(df_data_stg6.loc[(df_data_stg6["split"] == 1)])
        am[5][2] += len(df_data_stg6.loc[(df_data_stg6["split"] == 2)])
        am[5][3] += len(df_data_stg6.loc[(df_data_stg6["split"] == 3)])
        am[5][4] += len(df_data_stg6.loc[(df_data_stg6["split"] == 4)])
        am[5][5] += len(df_data_stg6.loc[(df_data_stg6["split"] == 5)])

        

    print("Preparing results..")
    print()
    # Find out the total amount of partitions per depth
    total = {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for key in total.keys():
        for key2 in am[0].keys():
            total[key] += am[key][key2]

    # Compute proportions per depth
    for key in am.keys():
        for key2 in pm[0].keys():
            pm[key][key2] = am[key][key2] / total[key]

    print("Results:")
    print()
    print("Amount of partitions per depth:", am)
    print()
    print("Proportion of partitions per depth:", pm)

    return pm, am


def lst2csv(lst, name_of_file):
    """!
    @brief Converts list of dictionaries to csv file

    @param [in] lst: List of dictionaries
    @param [in] name_of_file: Name to be given to the csv file
    """

    # Get all the fields that each dictionary has
    field_names = lst[0].keys()

    # Save to a csv file
    with open(name_of_file + '.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(lst)



def get_some_data_equaly(X, path_dir_l, classes, split_pos):
    """!
    Gets X amount of data from files
    """
    print("Active function: get_some_data")

    # Create new dir to save data
    new_dir = path_dir_l + "/some_data/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        data_size = len(orig_list)

        # Verbose
        print("Processing:", lbls_path)

        for c in range(classes):
            # Loop entries
            for k in range(X):

                try:
                    if orig_list[k][split_pos] == c:
                        # Save entry in final list
                        mod_list.append(orig_list[k])
                except:
                    break

            utils.echo("Complete: {per:.0%}".format(per=k/data_size))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4])
        lst2file(mod_list, new_path)


def lst2csv_v2(lst_lst, n_file, n_fields):
    """!
    @brief Converts list to csv file using panda dataframe

    @param [in] lst: List of lists
    @param [in] n_file: Name to be given to the csv file
    @param [in] n_fields: List of names given to each field
    """

    df = pd.DataFrame(lst_lst).transpose()
    df.columns = n_fields
    df.to_csv(n_file+".csv")     

def csv2lst(csv_file):
    """!
    @brief Reads csv file

    @param [in] csv_file: Path with the csv file
    @param [out] lst: List of dictionaries with the labels from the csv file
    """

    lst = pd.read_csv(csv_file)

    return lst

def file2lst(file):
    """!
    @brief Reads file

    @param [in] file: Path with the txt file
    @param [out] lst: List of dictionaries with the labels from a pickle file
    """

    with open(file+".txt", "rb") as fp:  # Pickling
        lst = pickle.load(fp)

    return lst

def lst2file(lst, name_of_file):
    """!
    @brief Converts list of dictionaries to file

    @param [in] lst: List of dictionaries
    @param [in] name_of_file: Name to be given to the file
    """

    # Save to file
    with open(name_of_file+".txt", "wb") as fp:  # Pickling
        pickle.dump(lst, fp)

def unite_labels_v6(dir_path_l, n_output_file="labels_pickle", color_ch=0):
    """!
    @brief Unites all the labels into a giant list. This version, follows a stage oriented approach. Uses just the
           specified color channel

    @param [in] dir_path_l: Path with all the labels (.dat files)
    @param [in] n_output_file: Name for the output file
    @param [in] color_ch: Color channel
    """

    # Obtain the name of the files
    files_list = get_files_from_folder(dir_path_l, ".dat")

    # New directory path
    output_dir = os.path.join(dir_path_l, n_output_file)
    create_dir(output_dir)

    # Move current process to the to the specified directory
    os.chdir(dir_path_l)

    # Create dict for files
    files_dict = {}
    for file in files_list:
        mini_dict = {}
        # Add stats
        num_records, file_size = file_stats(file)
        mini_dict["num_records"] = num_records
        # Add file object
        f = open(file, "rb")  # Opening file in binary mode read and write
        f_name = file[:-4]  # Get File name
        mini_dict["f"] = f
        mini_dict["f_name"] = f_name

        # Add to final dict
        files_dict[file] = mini_dict

    # File counter
    for file in files_dict.keys():
        # Amount of records per label
        step = files_dict[file]["num_records"]

        # Get file info from dict
        f = files_dict[file]["f"]
        f_name = files_dict[file]["f_name"]

        # Process all data from files
        transform_create_struct_faster_v2_mod_divs(f, f_name, step, output_dir, n_output_file, color_ch=color_ch)

def unite_labels_v6_mod(dir_path_l, n_output_file="labels_pickle", color_ch=0):
    """!
    @brief Unites all the labels into a giant list. This version, follows a stage oriented approach. Uses just the
           specified color channel

    @param [in] dir_path_l: Path with all the labels (.dat files)
    @param [in] n_output_file: Name for the output file
    @param [in] color_ch: Color channel
    """

    # Obtain the name of the files
    files_list = get_files_from_folder(dir_path_l, ".dat")

    # New directory path
    output_dir = os.path.join(dir_path_l, n_output_file)
    create_dir(output_dir)

    # Move current process to the to the specified directory
    os.chdir(dir_path_l)

    # Create dict for files
    files_dict = {}
    for file in files_list:
        mini_dict = {}
        # Add stats
        num_records, file_size = file_stats(file)
        mini_dict["num_records"] = num_records
        # Add file object
        f = open(file, "rb")  # Opening file in binary mode read and write
        f_name = file[:-4]  # Get File name
        mini_dict["f"] = f
        mini_dict["f_name"] = f_name

        # Add to final dict
        files_dict[file] = mini_dict

    # File counter
    for file in files_dict.keys():
        # Amount of records per label
        step = files_dict[file]["num_records"]

        # Get file info from dict
        f = files_dict[file]["f"]
        f_name = files_dict[file]["f_name"]

        # Process all data from files
        transform_create_struct_faster_v3(f, f_name, step, output_dir, n_output_file, color_ch=color_ch)

def create_dir(output_dir):
    """!
    @brief Creates a directory. If the directory already exists, it will be deleted

    @param [in] output_dir: Name of the directory
    """
    # Create new directory
    try:
        os.mkdir(output_dir)

    except:
        # Delete the directory
        shutil.rmtree(output_dir)
        # Recreate it
        os.mkdir(output_dir)

def labels_with_specific_cch(dir_path, cch=0):
    """!
    @brief Obtain from a group of labels in a pickle file the CUs which the color channel is 'cch'

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] cch: Color Channel
    """
    assert cch <=1

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = 'luma' if cch==0 else 'chroma'
    path = os.path.join(dir_path, folder_name)  # Compose path
    os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        for la in labels:
            if la["stg_1"]["color_ch"] == cch:
                new_labels.append(la)

        # Save label
        os.chdir(path)  # Change to the folder to save the data
        lst2file(new_labels, f[:-4] + "_cch" + str(cch))
        os.chdir(dir_path)  # Change back to the previous dir
        # Reset variable
        new_labels = []

def read_from_records_v2(f, f_name, num_records):
    """!
    @brief Read the information/file generated by the encoder. This version contains the file object. Adapted for the unite_labels_v3 function
    Dictionary containing all the info about the file: It's a dictionary of picture numbers, which then leads to a dictionary of the info.
    For example: records = {"Pic_0" :{"CU_0": {"colorChannel": 1,
                                               "CULoc_left": 2,
                                                ...
                                             }
                                     ...
                           ...      }
                           }

    @param [in] f: File object
    @param [in] f_name: Path where the file is located
    @param [out] num_records: Dictionary containing the information of all records
    """

    records = {}  # Dictionary with all the information extracted from the file
    CU_Counter = {}  # Dictionary keeping track of the number of CUs per Picture

    # Extract records
    for i in range(0, num_records):
        content = extract_content(f)
        pic_key = f_name + "_Pic_" + str(content["POC"])  # Key for the records and CU_counter dictionary
        # Update CU counter
        try:
            CU_Counter[pic_key] = CU_Counter[pic_key] + 1  # In case pic_key is not yet in the dictionary...
        except:
            # ... add it
            CU_Counter[pic_key] = 1

        CU_key = "CU_" + str(CU_Counter[pic_key] - 1)  # Key for the records dictionary
        # Store record
        try:
            # In case CU_key is not yet associated with a dictionary inside of the dictionary records...
            records[pic_key][CU_key] = {"POC": content["POC"], "color_ch": content["color_ch"],
                                        "CU_loc_left": content["CU_loc_left"],
                                        "CU_loc_top": content["CU_loc_top"], "CU_w": content["CU_w"],
                                        "CU_h": content["CU_h"],
                                        "RD_mode0": content["RD_mode0"], "RD_mode1": content["RD_mode1"],
                                        "RD_mode2": content["RD_mode2"], "RD_mode3": content["RD_mode3"],
                                        "RD_mode4": content["RD_mode4"],
                                        "RD_mode5": content["RD_mode5"]}

        except:
            # ... Associate pic_key with a dictionary, and then add CU_key and item with the dictionary created
            records[pic_key] = {}
            records[pic_key][CU_key] = {"POC": content["POC"], "color_ch": content["color_ch"],
                                        "CU_loc_left": content["CU_loc_left"],
                                        "CU_loc_top": content["CU_loc_top"], "CU_w": content["CU_w"],
                                        "CU_h": content["CU_h"],
                                        "RD_mode0": content["RD_mode0"], "RD_mode1": content["RD_mode1"],
                                        "RD_mode2": content["RD_mode2"], "RD_mode3": content["RD_mode3"],
                                        "RD_mode4": content["RD_mode4"],
                                        "RD_mode5": content["RD_mode5"]}

    return records, CU_Counter

def file_stats_v2(path):
    """!
    @brief Finds out the size of all binary files, computes the total amount of records, computes the amount of each CU

    @param [in] path: Path where the binary files are located
    @param [out] num_records: Number of records that all binary files contains
    @param [out] amount_dic: Dictionary with the amount of each CU
                            amount_dic = {"file_name": {"128x128L":100, "128x128C":100, ... , "4x4C", "4x4L"}, ...,
                                          "file_name2":{...}}, in which C stands for chroma and L for Luma
    @param [out] summary_dic: Dictionary with the sum of each CU type
    """

    # Initialize variable
    amount_dic = {}  # Dictionary with all the amounts information

    # Compute amount of each CU type
    # Get information about all the files
    files = get_files_from_folder(path, endswith='.dat')
    # Change process directory
    os.chdir(path)
    # Get info
    for name in files:
        # Data from file
        num_records, file_size = file_stats(name)

        # Reading and extracting information from file
        f = open(name, "rb")  # Opening file in binary mode read and write
        f_name = name[:-4]  # Get File name
        amount_dic[f_name] = {} # Initialize dict with the name of the file

        # Extract records
        for i in range(0, num_records):
            # CU info
            content = extract_content(f)

            # Figure out the color channel
            if content["color_ch"] == 0:
                color_ch = "L"
            else:
                color_ch = "C"

            # Obtain key
            if content["CU_w"] > content["CU_h"]:
                key = str(content["CU_w"]) + "x" + str(content["CU_h"]) + color_ch
            else:
                key = str(content["CU_h"]) + "x" + str(content["CU_w"]) + color_ch

            try:
                amount_dic[f_name][key] = amount_dic[f_name][key] + 1

            except:
                amount_dic[f_name][key] = 1

        # Print information
        print("File " + f_name + " has:")
        for key in amount_dic[f_name].keys():
            print("*", amount_dic[f_name][key], "of", key)

        print()
        f.close()  # Detach file

    # Print summary of all information
    summary_dic = {}
    for key in amount_dic.keys():
        for key2 in amount_dic[key].keys():
            try:
                summary_dic[key2] = amount_dic[key][key2] + summary_dic[key2]

            except:
                summary_dic[key2] = amount_dic[key][key2]

            try:
                summary_dic["total"] = amount_dic[key][key2] + summary_dic["total"]

            except:
                summary_dic["total"] = amount_dic[key][key2]
    print()
    print("In total the number of:")
    for key in summary_dic.keys():
        if key != "total":
            print("*", key, "is", summary_dic[key])

    print("And the total amount of CUs is:", summary_dic["total"])

    return summary_dic["total"], amount_dic, summary_dic


def compute_split_proportions(path, num_cus=float('inf')):
    """!
    @brief Compute the proportion of each split in the dataset

    @param [in] path: Path where the encoded data is located
    @param [in] num_cus: Number CUs to count for each file to calculate the proportions
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """

    # Initialize variables
    pm = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
    am = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    # Obtain the name of the files
    files_list = get_files_from_folder(path, ".dat")

    # Change dir workspace
    os.chdir(path)

    for name in files_list:
        # Read data from file
        num_records, file_size = file_stats(name)
        records, CU_Counter = read_from_records(name, num_records)

        # Process data
        lst_dicts = transform_raw_dataset(records)

        # Count
        count = 0
        for dic in lst_dicts:
            am[dic['split']] += 1

            # Total amount of data to search
            count += 1
            if count == num_cus:
                break

    # Find out the total amount of cus
    total = 0
    for val in am.values():
        total += val

    # Compute proportions
    for key in am.keys():
        pm[key] = am[key] / total

    print("Amount of each split:", am)
    print()
    print("Proportion of each split:", pm)

    return pm, am

def compute_split_proportions_with_custom_data(custom_dataset, stage, num_cus=float('inf')):
    """!
    @brief Compute the proportion of each split in the dataset (Custom dataset classs)

    @param [in] custom_dataset: Object with custom dataset
    @param [in] stage: Stage number that the proportions will be computed
    @param [in] num_cus: Number CUs to count to calculate the proportions
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """

    # Initialize variables
    pm = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
    am = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    for i in range(len(custom_dataset)):

        # Obtain sample from dataset
        sample = custom_dataset[i]
        # Best split
        try:
            split = sample['stg_'+str(stage)]['split']
            # Count
            am[split] += 1
        except:
            pass

        # Total amount of data to search
        if i == num_cus:
            break

    # Find out the total amount of cus
    total = 0
    for val in am.values():
        total += val

    # Compute proportions
    for key in am.keys():
        pm[key] = am[key] / total

    print("Amount of each split:", am)
    print()
    print("Proportion of each split:", pm)

    return pm, am

def compute_split_proportions_with_custom_data_multi(custom_dataset, split_pos_in_struct, num_cus=float('inf')):
    """!
    @brief Compute the proportion of each split in the dataset (Custom dataset classs)

    @param [in] custom_dataset: Object with custom dataset
    @param [in] stage: Stage number that the proportions will be computed
    @param [in] split_pos_in_struct: Position in dataset with the split information
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """

    # Initialize variables
    pm = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
    am = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    for i in range(len(custom_dataset)):

        # Obtain sample from dataset
        sample = custom_dataset[i]
        # Best split
        try:
            split = sample[split_pos_in_struct]

            # Count
            for s in split:
                am[s.item()] += 1
        except:
            pass

        # Total amount of data to search
        if i == num_cus:
            break

    # Find out the total amount of cus
    total = 0
    for val in am.values():
        total += val

    # Compute proportions
    for key in am.keys():
        pm[key] = am[key] / total

    print("Amount of each split:", am)
    print()
    print("Proportion of each split:", pm)

    return pm, am

    
def compute_split_proportions_with_path_multi_new(path, split_pos_in_struct, num_cus=float('inf')):
    """!
    @brief Compute the proportion of each split in the dataset (Custom dataset classs)

    @param [in] path: 
    @param [in] stage: Stage number that the proportions will be computed
    @param [in] split_pos_in_struct: Position in dataset with the split information
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """

    # Initialize variables
    pm = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
    am = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    # Obtain the name of the files
    files_list = get_files_from_folder(path, ".txt")

    # Change dir workspace
    os.chdir(path)

    for name in files_list:
        # Read data from file
        lst_lbls = file2lst(name[:-4])


        for k in range(len(lst_lbls)):
            # Obtain sample from dataset
            sample = lst_lbls[k]
            # Best split
            split = sample[split_pos_in_struct]
            am[split] += 1

            # Total amount of data to search
            if k == num_cus:
                break

    # Find out the total amount of cus
    total = 0
    for val in am.values():
        total += val

    # Compute proportions
    for key in am.keys():
        pm[key] = am[key] / total

    print("Amount of each split:", am)
    print()
    print("Proportion of each split:", pm)

    return pm, am


def compute_split_proportions_with_custom_data_multi_new(custom_dataset, split_pos_in_struct, num_cus=float('inf')):
    """!
    @brief Compute the proportion of each split in the dataset (Custom dataset classs)

    @param [in] custom_dataset: Object with custom dataset
    @param [in] stage: Stage number that the proportions will be computed
    @param [in] split_pos_in_struct: Position in dataset with the split information
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """

    # Initialize variables
    pm = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
    am = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    for i in range(len(custom_dataset)):

        # Obtain sample from dataset
        sample = custom_dataset[i]
        # Best split
        split = sample[split_pos_in_struct]
        am[split] += 1

        # Total amount of data to search
        if i == num_cus:
            break

    # Find out the total amount of cus
    total = 0
    for val in am.values():
        total += val

    # Compute proportions
    for key in am.keys():
        pm[key] = am[key] / total

    print("Amount of each split:", am)
    print()
    print("Proportion of each split:", pm)

    return pm, am


def compute_split_proportions_labels(path, num_cus=float('inf')):
    """!
    @brief Compute the proportion of each split in the dataset. This version receives a path with labels already processed

    @param [in] path: Path where the encoded data is located
    @param [in] num_cus: Number CUs to count for each file to calculate the proportions
    @param [out] pm: Dictionary with the proportion of each split. {0: 0.1, 1:0.01, ... , 5:0.3}
    @param [out] am: Dictionary with the amount of each split. {0: 10, 1:1, ... , 5:30}
    """

    # Initialize variables
    pm = {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0}
    am = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    # Obtain the name of the files
    files_list = get_files_from_folder(path, ".txt")

    # Change dir workspace
    os.chdir(path)

    for name in files_list:
        # Read data from file
        lst_lbls = file2lst(name[:-4])

        # Count
        count = 0
        for k in range(len(lst_lbls)):
            am[lst_lbls[k]['split']] += 1

            # Total amount of data to search
            count += 1
            if count == num_cus:
                break

    # Find out the total amount of cus
    total = 0
    for val in am.values():
        total += val

    # Compute proportions
    for key in am.keys():
        pm[key] = am[key] / total

    print("Amount of each split:", am)
    print()
    print("Proportion of each split:", pm)

    return pm, am



def balance_dataset(dir_path, stg, n_classes=6):
    """!
    @brief Balance dataset so that the number of the classes are the same

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] stg: Stage number
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels"
    path = os.path.join(dir_path, folder_name)  # Compose path
    os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    idx_classes = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        # Cu counter
        total_number_cu = len(labels)

        for k in range(len(labels)):
            class_to_add = k % n_classes  # Compute the class to add

            for i in range(len(labels)):
                idx = (i + idx_classes[class_to_add]) % total_number_cu

                if labels[idx]["stg_"+str(stg)]["split"] == class_to_add:
                    new_labels.append(labels[idx])
                    idx_classes[class_to_add] = idx + 1
                    break

        # Save label
        os.chdir(path)  # Change to the folder to save the data
        lst2file(new_labels, f[:-4] + folder_name)
        os.chdir(dir_path)  # Change back to the previous dir
        # Reset variable
        new_labels = []
        idx_classes = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}


def balance_dataset_JF(dir_path, n_classes=6):
    """!
    @brief Balance dataset so that the number of the classes are the same. Uses upsampling. Follows same strategy as
           the balance dataset function.

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels"
    path = os.path.join(dir_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    idx_classes = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        # Cu counter
        total_number_cu = len(labels)

        for k in range(len(labels)):
            class_to_add = k % n_classes  # Compute the class to add

            for i in range(len(labels)):
                idx = (i + idx_classes[class_to_add]) % total_number_cu

                if labels[idx]["split"] == class_to_add:
                    new_labels.append(labels[idx])
                    idx_classes[class_to_add] = idx + 1
                    break

        # Save label
        os.chdir(path)  # Change to the folder to save the data
        lst2file(new_labels, f[:-4] + folder_name)
        os.chdir(dir_path)  # Change back to the previous dir

        # Reset variables
        new_labels = []
        lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}


def balance_dataset_down(dir_path, n_classes=6):
    """!
    @brief Balance dataset so that the number of the classes are the same. Uses downsampling. Different strategy that
           of the balance_dataset function.

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels_downsamp"
    path = os.path.join(dir_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        for n in range(n_classes):
            class_to_add = n  

            # Loop labels
            for k in range(len(labels)):

                if labels[k]["split"] == class_to_add: # Find right class

                    lst_classes[class_to_add].append(labels[k]) # Add class

        # Remove empty lists
        rm_lst = []
        for key in lst_classes.keys():

            if len(lst_classes[key]) == 0:

                rm_lst.append(key)

        for key in rm_lst:
            
            del lst_classes[key] # Remove


        # Get the size of the smallest list"
        min_size = min(list(map(lambda x: len(x), lst_classes.values())))     

        for key in lst_classes.keys():

            new_labels.extend(lst_classes[key][:min_size])

        # Save label
        os.chdir(path)  # Change to the folder to save the data
        lst2file(new_labels, f[:-4] + folder_name)
        os.chdir(dir_path)  # Change back to the previous dir
        # Reset variable
        new_labels = []
        lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}


def balance_dataset_down_v2(dir_path):
    """!
    @brief Balance dataset so that the number of the classes are the same. Uses downsampling. Different strategy that
           of the balance_dataset function. Faster version

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels_downsamp"
    path = os.path.join(dir_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        # Loop labels
        for k in range(len(labels)):

            lst_classes[labels[k]["split"]].append(labels[k]) # Add class

    # Remove empty lists
    rm_lst = []
    for key in lst_classes.keys():

        if len(lst_classes[key]) == 0:

            rm_lst.append(key)

    for key in rm_lst:
        
        del lst_classes[key] # Remove

    # Get the size of the smallest list"
    min_size = min(list(map(lambda x: len(x), lst_classes.values())))     

    for key in lst_classes.keys():

        new_labels.extend(lst_classes[key][:min_size])

    # Save label
    os.chdir(path)  # Change to the folder to save the data
    lst2file(new_labels, f[:-4] + folder_name)
    os.chdir(dir_path)  # Change back to the previous dir
    # Reset variable
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}


def balance_dataset_down_v3(dir_path):
    """!
    @brief Balance dataset so that the number of the classes are the same. Uses downsampling. Different strategy that
           of the balance_dataset function. Faster version

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels_downsamp"
    path = os.path.join(dir_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        # Loop labels
        for k in range(len(labels)):

            lst_classes[labels[k]["split"]].append(labels[k]) # Add class

        # Remove empty lists
        rm_lst = []
        for key in lst_classes.keys():

            if len(lst_classes[key]) == 0:

                rm_lst.append(key)

        for key in rm_lst:
            
            del lst_classes[key] # Remove

        # Get the size of the smallest list"
        min_size = min(list(map(lambda x: len(x), lst_classes.values())))     

        for key in lst_classes.keys():

            new_labels.extend(lst_classes[key][:min_size])

        # Save label
        os.chdir(path)  # Change to the folder to save the data
        lst2file(new_labels, f[:-4] + folder_name)
        os.chdir(dir_path)  # Change back to the previous dir
        # Reset variable
        new_labels = []
        lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}


def balance_dataset_down_v4(dir_path):
    """!
    @brief Balance dataset so that the number of the classes are the same. Uses downsampling. Different strategy that
           of the balance_dataset function. Faster version. No dicts version

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels_downsamp"
    path = os.path.join(dir_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries
        # Loop labels
        for k in range(len(labels)):
            lst_classes[labels[k][-3]].append(labels[k]) # Add class

        # Remove empty lists
        rm_lst = []
        for key in lst_classes.keys():
            if len(lst_classes[key]) == 0:
                rm_lst.append(key)

        for key in rm_lst:
            del lst_classes[key] # Remove

        # Get the size of the smallest list"
        min_size = min(list(map(lambda x: len(x), lst_classes.values())))     

        for key in lst_classes.keys():
            new_labels.extend(lst_classes[key][:min_size])

        # Save label
        os.chdir(path)  # Change to the folder to save the data
        lst2file(new_labels, f[:-4] + folder_name)
        os.chdir(dir_path)  # Change back to the previous dir
        # Reset variable
        new_labels = []
        lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}


def balance_dataset_up(dir_path, n_classes=6):
    """!
    @brief Balance dataset so that the number of the classes are the same. Uses upsampling. Different strategy that
           of the balance_dataset function.

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels_upsamp"
    path = os.path.join(dir_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        for n in range(n_classes):
            class_to_add = n  

            # Loop labels
            for k in range(len(labels)):

                if labels[k]["split"] == class_to_add: # Find right class

                    lst_classes[class_to_add].append(labels[k]) # Add class

        # Remove empty lists
        rm_lst = []
        for key in lst_classes.keys():
            if len(lst_classes[key]) == 0:
                rm_lst.append(key)

        for key in rm_lst:
            del lst_classes[key] # Remove

        # Get the size of the smallest list"
        max_size = max(list(map(lambda x: len(x), lst_classes.values())))     

        # Extend final list with list of classes
        for key in lst_classes.keys():
            size_lst = len(lst_classes[key])

            if size_lst == max_size:
                new_labels.extend(lst_classes[key])
            
            else: # Upsample
                diff_size = max_size-size_lst 
                idx = 0

                for n in range(diff_size):
                    lst_classes[key].append(lst_classes[key][idx])
                    idx = (idx + 1) % size_lst

                new_labels.extend(lst_classes[key])

        # Save label
        os.chdir(path)  # Change to the folder to save the data
        lst2file(new_labels, f[:-4] + folder_name)
        os.chdir(dir_path)  # Change back to the previous dir
        # Reset variable
        new_labels = []
        lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}


def balance_dataset_up_v2(dir_path):
    """!
    @brief Balance dataset so that the number of the classes are the same. Uses upsampling. Different strategy that
           of the balance_dataset function. Faster version

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels_upsamp"
    path = os.path.join(dir_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        # Loop labels
        for k in range(len(labels)):

            lst_classes[labels[k]["split"]].append(labels[k]) # Add class

    # Remove empty lists
    rm_lst = []
    for key in lst_classes.keys():

        if len(lst_classes[key]) == 0:

            rm_lst.append(key)

    for key in rm_lst:
        
        del lst_classes[key]  # Remove

    # Get the size of the largest list
    max_size = max(list(map(lambda x: len(x), lst_classes.values())))     

    # Extend final list with list of classes
    for key in lst_classes.keys():
        size_lst = len(lst_classes[key])

        if size_lst == max_size:
            new_labels.extend(lst_classes[key])
        
        else: # Upsample
            diff_size = max_size-size_lst 
            idx = 0

            for n in range(diff_size):
                lst_classes[key].append(lst_classes[key][idx])
                idx = (idx + 1) % size_lst

            new_labels.extend(lst_classes[key])

    # Save label
    os.chdir(path)  # Change to the folder to save the data
    lst2file(new_labels, f[:-4] + folder_name)
    os.chdir(dir_path)  # Change back to the previous dir
    # Reset variable
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}


def balance_dataset_up_v3(dir_path):
    """!
    @brief Balance dataset so that the number of the classes are the same. Uses upsampling. Different strategy that
           of the balance_dataset function. Faster version

    @param [in] dir_path: Path with all the labels (.txt files)
    @param [in] n_classes: Number of classes to try to balance
    """

    # Move current process to the Labels directory
    os.chdir(dir_path)

    # Create folder to save the files
    folder_name = "balanced_labels_upsamp"
    path = os.path.join(dir_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(dir_path, endswith=".txt")
    new_labels = []
    lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    # Iterate all the files
    for f in files:
        # Obtain the labels
        labels = file2lst(f[:-4]) # List of dictionaries

        # Loop labels
        for k in range(len(labels)):

            lst_classes[labels[k]["split"]].append(labels[k]) # Add class

        # Remove empty lists
        rm_lst = []
        for key in lst_classes.keys():

            if len(lst_classes[key]) == 0:

                rm_lst.append(key)

        for key in rm_lst:
            
            del lst_classes[key]  # Remove

        # Get the size of the largest list
        max_size = max(list(map(lambda x: len(x), lst_classes.values())))     

        # Extend final list with list of classes
        for key in lst_classes.keys():
            size_lst = len(lst_classes[key])

            if size_lst == max_size:
                new_labels.extend(lst_classes[key])
            
            else: # Upsample
                diff_size = max_size-size_lst 
                idx = 0

                for n in range(diff_size):
                    lst_classes[key].append(lst_classes[key][idx])
                    idx = (idx + 1) % size_lst

                new_labels.extend(lst_classes[key])

        # Save label
        os.chdir(path)  # Change to the folder to save the data
        lst2file(new_labels, f[:-4] + folder_name)
        os.chdir(dir_path)  # Change back to the previous dir
        # Reset variable
        new_labels = []
        lst_classes = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}


def gen_dataset_types(d_path, valid_percent):
    """!
    @brief Generate a dataset for trainign, validating and testing. This is done by concatenating all of the data from a folder and then
           dividing it in 3 parts

    @param [in] d_path: Path with all the labels (.txt files)
    @param [in] valid_percent: Percentage of data allocated to test and validation data
    """

    # Move current process to the Labels directory
    os.chdir(d_path)

    # Create folder to save the files
    folder_name = "train_valid_test"
    path = os.path.join(d_path, folder_name)  # Compose path
    try:
        os.mkdir(path)  # Create folder

    except:
        shutil.rmtree(path)  # Remove existing folder
        os.mkdir(path)  # Create folder

    # Initialize variables
    files = get_files_from_folder(d_path, endswith=".txt")

    # Retrieve data from files
    lst_data = []
    for f in files:
        labels = file2lst(f[:-4]) # List of dictionaries
        lst_data.extend(labels)

    # Divide data
    train, valid = train_test_split(lst_data, test_size=valid_percent)
    test_percent = (valid_percent)/(1-valid_percent)
    train, test = train_test_split(train, test_size=test_percent)

    # Save data
    p_train = os.path.join(path, "train")  # Path to data
    lst2file(train, p_train)
    p_valid = os.path.join(path, "valid")  # Path to data
    lst2file(valid, p_valid)
    p_test = os.path.join(path, "test")  # Path to data
    lst2file(test, p_test)



def change_struct_64x64_eval(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")


    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
            lst2file(mod_list, new_path)


def change_struct_32x32_eval(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)

def change_struct_64x64(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)


def change_struct_64x64_no_dupl_v3(path_dir_l):
    """!
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

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        if type(row["stg_2"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

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
    orig_list = file2lst(lbls_path[:-4])
    data_size = len(orig_list)
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
    lst2file(mod_list, new_path)


def change_struct_64x64_no_dupl_v2(path_dir_l):
    """!
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

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_2"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        data_size = len(orig_list)
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
        lst2file(mod_list, new_path)


def change_struct_32x32(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)

def change_struct_32x32_no_dupl(path_dir_l):
    """!
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

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)


def change_struct_32x32_no_dupl_v2(path_dir_l):
    """!
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

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_3"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        data_size = len(orig_list)
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
        lst2file(mod_list, new_path)


def change_struct_32x32_no_dupl_v3(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

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
    orig_list = file2lst(lbls_path[:-4])
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
    lst2file(mod_list, new_path)


def change_struct_32x32_no_dupl_v2_test(path_dir_l):
    """!
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

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_3"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)

def change_struct_16x16_no_dupl_v2(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)

def list2tuple(l):
    return tuple(list2tuple(x) for x in l) if type(l) is list else l


def tuple2list(l):
    return list(tuple2list(x) for x in l) if type(l) is tuple else l


def change_struct_8x8_no_dupl_v2(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg6_v4(path_dir_l):
    """!
    This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 6
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg6_v4")
    t0 = time.time()

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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg5_v4(path_dir_l):
    """!
    This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 5
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg5_v4")
    t0 = time.time()

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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg2_v4(path_dir_l):
    """!
    This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 2
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg2_v4")
    t0 = time.time()

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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)

def change_struct_no_dupl_stg4_v4(path_dir_l):
    """!
    This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 4
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg4_v4")
    t0 = time.time()

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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg3_v4(path_dir_l):
    """!
    This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stage 3
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg3_v4")
    t0 = time.time()

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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_32x16_no_dupl_v2(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_32x8_no_dupl_v2(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        data_size = len(orig_list)
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
        lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)

def change_struct_16x8_no_dupl_v2(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        data_size = len(orig_list)
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
        lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_8x4_no_dupl_v2(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        data_size = len(orig_list)
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
        lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_32x4_no_dupl_v2(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        data_size = len(orig_list)
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
        lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_16x4_no_dupl_v2(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        data_size = len(orig_list)
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
        lst2file(mod_list, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_16x16_no_dupl_v3(path_dir_l):
    """!
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

    # List with columns
    columns = ["POC", "pic_name", "cu_pos_x", "cu_pos_y", "cu_size_w", "cu_size_h", "split"]

    def right_rows(row):
        
        if type(row["stg_4"]) != list:
            return True
        else:
            return False

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

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
    orig_list = file2lst(lbls_path[:-4])
    data_size = len(orig_list)
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
    lst2file(mod_list, new_path)


def change_struct_16x16(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # List to save entries
        mod_list = []

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(mod_list, new_path)


def change_struct_no_dupl_stg_4_complexity_v4(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg_3_complexity_v4(path_dir_l):
    """!
        This version is like the change_struct_32x32_no_dupl_v2, but it is applied to stages 3. Here it is going to be obtained data to be used for the complexity assesment
    """
    # Initial steps
    print("Active function: change_struct_no_dupl_stg_3_complexity_v4")
    t0 = time.time()

    def list2tuple(l):
        return tuple(list2tuple(x) for x in l) if type(l) is list else l

    def tuple2list(l):
        return list(tuple2list(x) for x in l) if type(l) is tuple else l

    # Create new dir to save data
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg_3_compl_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
        orig_list = [d for d in orig_list if type(d['stg_3']) != list]  # Remove empty stage

        # Verbose
        print("Processing:", lbls_path)

        # Create appropriate structure
        my_set = {list2tuple( [[l["stg_3"]["RD0"], l["stg_3"]['RD1'], l["stg_3"]['RD2'], l["stg_3"]['RD3'], l["stg_3"]['RD4'], l["stg_3"]["RD5"]], \
                               [l["stg_2"]["cu_pos"]["CU_loc_top"] - l["stg_1"]["cu_pos"]["CU_loc_top"], l["stg_2"]["cu_pos"]["CU_loc_left"] - l["stg_1"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_3"]["cu_pos"]["CU_loc_top"] - l["stg_2"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"] - l["stg_2"]["cu_pos"]["CU_loc_left"]],\
                               [l["stg_2"]["cu_size"]["height"], l["stg_2"]["cu_size"]["width"]], \
                               [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]], \
                                l["stg_2"]["split"], l["stg_3"]["split"], \
                                l["stg_1"]["real_CTU"], l["stg_1"]["color_ch"], \
                                l["POC"], l["pic_name"], \
                                [l["stg_3"]["cu_pos"]["CU_loc_top"], l["stg_3"]["cu_pos"]["CU_loc_left"]], \
                                [l["stg_3"]["cu_size"]["height"], l["stg_3"]["cu_size"]["width"]] ]) for l in orig_list}  # Remove empty stage

        # Convert back to list
        my_set = tuple2list(tuple((my_set)))

        # Save list to file with the same name
        new_path = os.path.join(new_dir, f[:-4]+"mod_with_struct")
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg_2_complexity_v4(path_dir_l):
    """!
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
    new_dir = path_dir_l + "/mod_with_struct_change_no_dupl_stg_2_compl_v4/"
    try:
        os.mkdir(new_dir)
    except:
        # Delete the directory
        shutil.rmtree(new_dir)
        # Recreate it
        os.mkdir(new_dir)

    # Get files names from folder
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg_6_complexity_v4(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)


def change_struct_no_dupl_stg_5_complexity_v4(path_dir_l):
    """!
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
    files_l = get_files_from_folder(path_dir_l, endswith=".txt")

    my_set = set()

    for f in files_l:
        # Make labels path
        lbls_path = os.path.join(path_dir_l, f)

        # Read file
        orig_list = file2lst(lbls_path[:-4])
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
        lst2file(my_set, new_path)

    print("Time Elapsed:", time.time() - t0)

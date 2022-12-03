"""@package docstring 

@file CustomDataset.py 

@brief This library contains usefull functions to visualise data and also Classes to store and organise data structures. 
 
@section libraries_CustomDataset Libraries 
- os
- torch
- warnings
- numpy
- dataset_utils
- cv2
- random
- __future__
- torch.utils.data
- re

@section classes_CustomDataset Classes 
- CUDatasetBase
- CUDatasetStg4 
- CUDatasetStg5 
- CUDatasetStg6
- CUDatasetStg2Compl 
- CUDatasetStg6Compl 
- CUDatasetStg3Compl 
- CUDatasetStg4Compl
- CUDatasetStg5Compl
- SamplerStg4 
- SamplerStg5 
- SamplerStg6 

@section functions_CustomDataset Functions 
- yuv2bgr(matrix)
- bgr2yuv(matrix)
- get_cu_old(f_path, f_size, cu_pos, cu_size, frame_number)
- get_cu(f_path, f_size, cu_pos, cu_size, frame_number)
- get_file_size(name)
- resize(img, scale_percent)
- show_CU(image, cu_size, cu_pos)
- show_all_CUs(CUDataset, file_name, POC, cu_size)

@section global_vars_CustomDataset Global Variables 
- None 

@section todo_CustomDataset TODO 
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

@section author_CustomDataset Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2022-12-02 18:21:21.122664
"""

# ==============================================================
# Imports
# ==============================================================

from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import re
from dataset_utils import VideoCaptureYUV
import dataset_utils
import random
# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")


# ==============================================================
# Functions
# ==============================================================

def yuv2bgr(matrix):
    """!
    Converts yuv matrix to bgr matrix

    @param [in] matrix: Yuv matrix
    @param [out] bgr: Bgr conversion
    """

    # Convert from yuv to bgr
    bgr = cv2.cvtColor(matrix, cv2.COLOR_YUV2BGR_I420)  # cv2.COLOR_YUV2BGR_NV21)

    return bgr

def bgr2yuv(matrix):
    """!
    Converts BGR matrix to YUV matrix

    @param [in] matrix: BGR matrix
    @param [out] YUV: YUV conversion
    """

    # Convert from bgr to yuv
    YUV = cv2.cvtColor(matrix, cv2.COLOR_BGR2YUV_I420)  # cv2.COLOR_YUV2BGR_NV21)

    return YUV

def get_cu(f_path, f_size, cu_pos, cu_size, frame_number):
    """!
    Get CU from image

    @param [in] f_path: Path of file to get the CU from
    @param [in] f_size: YUV file dimensions (height, width)
    @param [in] cu_pos: Tuple with the position of the CU (y position (height), x position (width))
    @param [in] cu_size: Tuple with the size of the CU (y position (height), x position (width))
    @param [in] frame_number: Number of the frame containing the CU
    @param [out] CU: CU with all the information from all the components(Luminance and Chroma)
    @param [out] CU_Y: CU with the Luminance component
    @param [out] CU_U: CU with the Chroma (Blue) component
    @param [out] CU_V: CU with the Chroma (Red) component
    @param [out] frame_CU: specified frame that contains the CU
    """

    # Get file data
    yuv_file = VideoCaptureYUV(f_path, f_size)

    # Get the specific frame
    ret, yuv_frame, luma_frame, chr_u, chr_v = yuv_file.read_raw(frame_number)  # Read Frame from File

    # Return false in case a wrongful value is returned
    if not ret:
        return False

    # Get region that contain the CU
    CU_Y = luma_frame[cu_pos[0]: (cu_size[0] + cu_pos[0]), cu_pos[1]: (cu_size[1] + cu_pos[1])]  # CU Luma component

    # Get the CU different components
    CU_U = chr_u[int(cu_pos[0]/2): int((cu_size[0] + cu_pos[0])/2), int(cu_pos[1]/2): int((cu_size[1] + cu_pos[1])/2)]  # CU Chroma blue component
    CU_V = chr_v[int(cu_pos[0]/2): int((cu_size[0] + cu_pos[0])/2), int(cu_pos[1]/2): int((cu_size[1] + cu_pos[1])/2)]  # CU Chroma red component

    return yuv_frame, CU_Y, CU_U, CU_V

def get_file_size(name):
    """!
    Retrieves information about the YUV file info (width and height)

    @param [in] name: Name of the file where the file is located
    @param [out] file_info: Dictionary with information about the yuv file
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

def resize(img, scale_percent):
    """!
    Resizes a BGR image

    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def show_CU(image, cu_size, cu_pos):
    """!
    Shows CU in a image
    """

    # Convert image to bgr
    image = yuv2bgr(image)

    # Blue color in BGR
    color = (255, 0, 0)

    cu_pos_begin = (cu_pos[1], cu_pos[0])

    # Line thickness of 2 px
    thickness = 2

    # CU end position
    cu_pos_end = (cu_pos[1] + cu_size[1], cu_pos[0] + cu_size[0])

    # Draw a rectangle with blue line borders of thickness of 2 px
    image_final = cv2.rectangle(image, cu_pos_begin, cu_pos_end, color, thickness)
    resize_img = resize(image_final, 50)
    cv2.imshow('Frame', resize_img)

    # Press q on keyboard to  exit
    print('\n\n\n Press q to exit')
    while not (cv2.waitKey(100) & 0xFF == ord('q')):
        pass
    # Destroy all windows
    cv2.destroyAllWindows()


def show_all_CUs(CUDataset, file_name, POC, cu_size):
    """!
    Shows all CUs in a image with a specific size, TO BE USED WITH THE CU ORIENTE LABELS
    """
    image_obtained = False
    image = None
    for sample in CUDataset:
        # print(sample['file_name'], sample['POC'], sample['CU_size'])

        # Draw CU in image
        if sample['file_name'] == file_name and sample['POC'] == POC and sample['CU_size'] == cu_size:
            # print('hello2')
            f_size = get_file_size(file_name)  # Dict with the size of the image
            cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(sample['img_path'], (f_size['height'], f_size['width']),
                                                    sample['CU_pos'], sample['CU_size'], POC)

            if not image_obtained:
                # Convert image to bgr
                image = yuv2bgr(frame_CU)
                # Blue color in BGR
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 2
                # Change flag
                image_obtained = True

            # CU end position
            cu_pos_end = (sample['CU_pos'][0] + sample['CU_size'][0], sample['CU_pos'][1] + sample['CU_size'][1])

            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv2.rectangle(image, sample['CU_pos'], cu_pos_end, color, thickness)

    resize_img = resize(image, 50)
    cv2.imshow('Frame', resize_img)
    # Press Q on keyboard to  exit
    while not (cv2.waitKey(100) & 0xFF == ord('q')):
        pass
    # Destroy all windows
    cv2.destroyAllWindows()


# ==============================================================
# Classes
# ==============================================================

class CUDatasetBase(Dataset):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. Works for stage 2 and 3
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            @param files_path (string): Path to the files with annotations.
            @param channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """

        # Store file paths
        self.files_path = files_path
        self.files = dataset_utils.get_files_from_folder(self.files_path, endswith = ".txt")
        
        # Compute number of entries per file
        self.lst_entries_nums = self.obtain_files_sizes(self.files)
        
        # Obtain amount of entries in all dataset files
        self.total_num_entries = 0
        for f in self.lst_entries_nums:
            self.total_num_entries += f

        # Initialize variables
        self.channel = channel
        self.index_lims = []
        self.data_files = []

        # Compute index limits for each file
        for k in range(len(self.lst_entries_nums)):
            sum = -1
            for f in self.lst_entries_nums[:k+1]:
                sum += f
            self.index_lims.append(sum)

        # Dataset for each file
        for k in range(len(self.files)):
            self.data_files.append(dataset_utils.file2lst(self.files_path + "/" + self.files[k][0:-4]))

    def __len__(self):
        return self.total_num_entries

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.

            @return out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []

        # Add Luma
        CU_Y = entry["real_CTU"]

        # Add dimension
        CU_Y = torch.unsqueeze(CU_Y, 0)

        # Convert to float
        CU_Y = CU_Y.to(dtype=torch.float32)

        # CU positions within frame
        cu_pos = entry["cu_pos"]
        
        # CU positions within frame
        cu_size = entry["cu_size"]
        
        # Best split for CU
        split = entry["split"]

        # Rate distortion costs
        RDs = entry['RD']

        # Unite
        RDs = np.reshape(RDs, (-1, 6))

        # Save values
        info_lst.append(CU_Y)
        info_lst.append(cu_pos)
        info_lst.append(cu_size)
        info_lst.append(split)
        info_lst.append(RDs)

        return info_lst

    def obtain_files_sizes(self, files):
        """!
        Args:
            @param files (lst): List containing the names of files with CUs info
        """

        # Initialize variable
        lst = []

        # Create list with the number of entries of each file
        for f in files:
            f_path = self.files_path + "/" + f[0:-4]
            file_obj = dataset_utils.file2lst(f_path)
            num_entries = len(file_obj)
            lst.append(num_entries)

        return lst

    def select_entry(self, idx):
        """!
        Args:
            @param idx (int): Index with the position to search for a specific entry
        """

        for k in range(len(self.index_lims)):

            if idx <= self.index_lims[k]:

                # Obtain index in the object
                idx_obj = idx - (self.index_lims[k] - self.lst_entries_nums[k] + 1)

                file_obj = self.data_files[k]

                # Obtain entry
                entry = file_obj[idx_obj]

                return entry

        raise Exception("Entry not found!!")

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Select entry
        found = False
        while not found:
            entry = self.select_entry(idx)  # Choose entry

            ### Get info from label and process
            # Get sample
            sample = self.get_sample(entry)

            ctu_size = (sample[0].shape[-2], sample[0].shape[-1])
            if (128, 128) == ctu_size:
                found = True

            else:
                idx = (idx + 1) % self.total_num_entries  # Increment index until a acceptable entry is found

        return sample

class CUDatasetStg4(CUDatasetBase):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. For stage 4
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            @param files_path (string): Path to the files with annotations.
            @param channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg4, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.

            @return out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        color_ch = entry[-1]

        if color_ch == self.channel:
            # Add Real CU
            CU_Y = entry[-2]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = torch.reshape(torch.tensor(entry[3]), (-1, 2))
            cu_pos_stg3 = torch.reshape(torch.tensor(entry[2]), (-1, 2))
            cu_pos_stg2 = torch.reshape(torch.tensor(entry[1]), (-1, 2))

            # CU sizes within frame
            cu_size = torch.reshape(torch.tensor(entry[6]), (-1, 2))
            cu_size_stg3 = torch.reshape(torch.tensor(entry[5]), (-1, 2))
            cu_size_stg2 = torch.reshape(torch.tensor(entry[4]), (-1, 2))

            # Best split for CU
            split = entry[9]
            split_stg3 = entry[8]
            split_stg2 = entry[7]

            # Rate distortion costs
            RDs = torch.reshape(torch.tensor(entry[0]), (-1, 6))

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos_stg2)
            info_lst.append(cu_pos_stg3)
            info_lst.append(cu_pos)
            info_lst.append(cu_size_stg2)
            info_lst.append(cu_size_stg3)
            info_lst.append(cu_size)
            info_lst.append(split_stg2)
            info_lst.append(split_stg3)
            info_lst.append(split)
            info_lst.append(RDs)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class CUDatasetStg6(CUDatasetBase):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. For stage 6
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            @param files_path (string): Path to the files with annotations.
            @param channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg6, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.
        """

        # Initialize variable
        info_lst = []
        color_ch = entry[-1]

        if color_ch == self.channel:
            # Add Real CU
            CU_Y = entry[-2]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = torch.reshape(torch.tensor(entry[5]), (-1, 2))
            cu_pos_stg5 = torch.reshape(torch.tensor(entry[4]), (-1, 2))
            cu_pos_stg4 = torch.reshape(torch.tensor(entry[3]), (-1, 2))
            cu_pos_stg3 = torch.reshape(torch.tensor(entry[2]), (-1, 2))
            cu_pos_stg2 = torch.reshape(torch.tensor(entry[1]), (-1, 2))

            # CU sizes within frame
            cu_size = torch.reshape(torch.tensor(entry[10]), (-1, 2))
            cu_size_stg5 = torch.reshape(torch.tensor(entry[9]), (-1, 2))
            cu_size_stg4 = torch.reshape(torch.tensor(entry[8]), (-1, 2))
            cu_size_stg3 = torch.reshape(torch.tensor(entry[7]), (-1, 2))
            cu_size_stg2 = torch.reshape(torch.tensor(entry[6]), (-1, 2))

            # Best split for CU
            split = entry[15]
            split_stg5 = entry[14]
            split_stg4 = entry[13]
            split_stg3 = entry[12]
            split_stg2 = entry[11]

            # Rate distortion costs
            RDs = torch.reshape(torch.tensor(entry[0]), (-1, 6))

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos_stg2)
            info_lst.append(cu_pos_stg3)
            info_lst.append(cu_pos_stg4)
            info_lst.append(cu_pos_stg5)
            info_lst.append(cu_pos)
            info_lst.append(cu_size_stg2)
            info_lst.append(cu_size_stg3)
            info_lst.append(cu_size_stg4)
            info_lst.append(cu_size_stg5)
            info_lst.append(cu_size)
            info_lst.append(split_stg2)
            info_lst.append(split_stg3)
            info_lst.append(split_stg4)
            info_lst.append(split_stg5)
            info_lst.append(split)
            info_lst.append(RDs)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class CUDatasetStg5(CUDatasetBase):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. For stage 5
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            @param files_path (string): Path to the files with annotations.
            @param channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.

            @return out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        color_ch = entry[-1]

        if color_ch == self.channel:
            # Add Real CU
            CU_Y = entry[-2]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = torch.reshape(torch.tensor(entry[4]), (-1, 2))
            cu_pos_stg4 = torch.reshape(torch.tensor(entry[3]), (-1, 2))
            cu_pos_stg3 = torch.reshape(torch.tensor(entry[2]), (-1, 2))
            cu_pos_stg2 = torch.reshape(torch.tensor(entry[1]), (-1, 2))

            # CU sizes within frame
            cu_size = torch.reshape(torch.tensor(entry[8]), (-1, 2))
            cu_size_stg4 = torch.reshape(torch.tensor(entry[7]), (-1, 2))
            cu_size_stg3 = torch.reshape(torch.tensor(entry[6]), (-1, 2))
            cu_size_stg2 = torch.reshape(torch.tensor(entry[5]), (-1, 2))

            # Best split for CU
            split = entry[12]
            split_stg4 = entry[11]
            split_stg3 = entry[10]
            split_stg2 = entry[9]

            # Rate distortion costs
            RDs = torch.reshape(torch.tensor(entry[0]), (-1, 6))

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos_stg2)
            info_lst.append(cu_pos_stg3)
            info_lst.append(cu_pos_stg4)
            info_lst.append(cu_pos)
            info_lst.append(cu_size_stg2)
            info_lst.append(cu_size_stg3)
            info_lst.append(cu_size_stg4)
            info_lst.append(cu_size)
            info_lst.append(split_stg2)
            info_lst.append(split_stg3)
            info_lst.append(split_stg4)
            info_lst.append(split)
            info_lst.append(RDs)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class CUDatasetStg5Compl(CUDatasetBase):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. For stage 5
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            @param files_path (string): Path to the files with annotations.
            @param channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg5Compl, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.

            @return out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        color_ch = entry[14]

        if color_ch == self.channel:
            # Add Real CU
            CU_Y = entry[13]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = torch.reshape(torch.tensor(entry[4]), (-1, 2))
            cu_pos_stg4 = torch.reshape(torch.tensor(entry[3]), (-1, 2))
            cu_pos_stg3 = torch.reshape(torch.tensor(entry[2]), (-1, 2))
            cu_pos_stg2 = torch.reshape(torch.tensor(entry[1]), (-1, 2))

            # CU sizes within frame
            cu_size = torch.reshape(torch.tensor(entry[8]), (-1, 2))
            cu_size_stg4 = torch.reshape(torch.tensor(entry[7]), (-1, 2))
            cu_size_stg3 = torch.reshape(torch.tensor(entry[6]), (-1, 2))
            cu_size_stg2 = torch.reshape(torch.tensor(entry[5]), (-1, 2))

            # Best split for CU
            split = entry[12]
            split_stg4 = entry[11]
            split_stg3 = entry[10]
            split_stg2 = entry[9]

            # Other information
            POC = entry[15]
            pic_name = entry[16]
            orig_pos_x = entry[17][1]
            orig_pos_y = entry[17][0]
            orig_size_h = entry[18][1]
            orig_size_w = entry[18][0]

            # Rate distortion costs
            RDs = torch.reshape(torch.tensor(entry[0]), (-1, 6))

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos_stg2)
            info_lst.append(cu_pos_stg3)
            info_lst.append(cu_pos_stg4)
            info_lst.append(cu_pos)
            info_lst.append(cu_size_stg2)
            info_lst.append(cu_size_stg3)
            info_lst.append(cu_size_stg4)
            info_lst.append(cu_size)
            info_lst.append(split_stg2)
            info_lst.append(split_stg3)
            info_lst.append(split_stg4)
            info_lst.append(split)
            info_lst.append(RDs)
            # Other data
            info_lst.append(orig_pos_x)
            info_lst.append(orig_pos_y)
            info_lst.append(orig_size_h)
            info_lst.append(orig_size_w)
            info_lst.append(POC)
            info_lst.append(pic_name)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class CUDatasetStg2Compl(CUDatasetBase):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. For stage 2
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            @param files_path (string): Path to the files with annotations.
            @param channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg2Compl, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.

            @return out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        color_ch = entry[5]

        if color_ch == self.channel:
            # Add Real CU
            CU_Y = entry[4]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = torch.reshape(torch.tensor(entry[1]), (-1, 2))


            # CU sizes within frame
            cu_size = torch.reshape(torch.tensor(entry[2]), (-1, 2))


            # Best split for CU
            split = entry[3]

            # Rate distortion costs
            RDs = torch.reshape(torch.tensor(entry[0]), (-1, 6))

            # Other information
            POC = entry[6]
            pic_name = entry[7]
            orig_pos_x = entry[8][1]
            orig_pos_y = entry[8][0]
            orig_size_h = entry[9][1]
            orig_size_w = entry[9][0]

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos)
            info_lst.append(cu_size)
            info_lst.append(split)
            info_lst.append(RDs)
            # Other data
            info_lst.append(orig_pos_x)
            info_lst.append(orig_pos_y)
            info_lst.append(orig_size_h)
            info_lst.append(orig_size_w)
            info_lst.append(POC)
            info_lst.append(pic_name)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class CUDatasetStg6Compl(CUDatasetBase):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. For stage 6
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg6Compl, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.
        """

        # Initialize variable
        info_lst = []
        color_ch = entry[17]

        if color_ch == self.channel:
            # Add Real CU
            CU_Y = entry[16]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = torch.reshape(torch.tensor(entry[5]), (-1, 2))
            cu_pos_stg5 = torch.reshape(torch.tensor(entry[4]), (-1, 2))
            cu_pos_stg4 = torch.reshape(torch.tensor(entry[3]), (-1, 2))
            cu_pos_stg3 = torch.reshape(torch.tensor(entry[2]), (-1, 2))
            cu_pos_stg2 = torch.reshape(torch.tensor(entry[1]), (-1, 2))

            # CU sizes within frame
            cu_size = torch.reshape(torch.tensor(entry[10]), (-1, 2))
            cu_size_stg5 = torch.reshape(torch.tensor(entry[9]), (-1, 2))
            cu_size_stg4 = torch.reshape(torch.tensor(entry[8]), (-1, 2))
            cu_size_stg3 = torch.reshape(torch.tensor(entry[7]), (-1, 2))
            cu_size_stg2 = torch.reshape(torch.tensor(entry[6]), (-1, 2))

            # Best split for CU
            split = entry[15]
            split_stg5 = entry[14]
            split_stg4 = entry[13]
            split_stg3 = entry[12]
            split_stg2 = entry[11]

            # Rate distortion costs
            RDs = torch.reshape(torch.tensor(entry[0]), (-1, 6))

            # Other information
            POC = entry[18]
            pic_name = entry[19]
            orig_pos_x = entry[20][1]
            orig_pos_y = entry[20][0]
            orig_size_h = entry[21][1]
            orig_size_w = entry[21][0]

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos_stg2)
            info_lst.append(cu_pos_stg3)
            info_lst.append(cu_pos_stg4)
            info_lst.append(cu_pos_stg5)
            info_lst.append(cu_pos)
            info_lst.append(cu_size_stg2)
            info_lst.append(cu_size_stg3)
            info_lst.append(cu_size_stg4)
            info_lst.append(cu_size_stg5)
            info_lst.append(cu_size)
            info_lst.append(split_stg2)
            info_lst.append(split_stg3)
            info_lst.append(split_stg4)
            info_lst.append(split_stg5)
            info_lst.append(split)
            info_lst.append(RDs)
            # Other data
            info_lst.append(orig_pos_x)
            info_lst.append(orig_pos_y)
            info_lst.append(orig_size_h)
            info_lst.append(orig_size_w)
            info_lst.append(POC)
            info_lst.append(pic_name)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class CUDatasetStg4Compl(CUDatasetBase):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. For stage 4
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            @param files_path (string): Path to the files with annotations.
            @param channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg4Compl, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.

            @param out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        color_ch = entry[11]

        if color_ch == self.channel:
            # Add Real CU
            CU_Y = entry[10]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = torch.reshape(torch.tensor(entry[3]), (-1, 2))
            cu_pos_stg3 = torch.reshape(torch.tensor(entry[2]), (-1, 2))
            cu_pos_stg2 = torch.reshape(torch.tensor(entry[1]), (-1, 2))

            # CU sizes within frame
            cu_size = torch.reshape(torch.tensor(entry[6]), (-1, 2))
            cu_size_stg3 = torch.reshape(torch.tensor(entry[5]), (-1, 2))
            cu_size_stg2 = torch.reshape(torch.tensor(entry[4]), (-1, 2))

            # Best split for CU
            split = entry[9]
            split_stg3 = entry[8]
            split_stg2 = entry[7]

            # Rate distortion costs
            RDs = torch.reshape(torch.tensor(entry[0]), (-1, 6))

            # Other information
            POC = entry[12]
            pic_name = entry[13]
            orig_pos_x = entry[14][1]
            orig_pos_y = entry[14][0]
            orig_size_h = entry[15][1]
            orig_size_w = entry[15][0]

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos_stg2)
            info_lst.append(cu_pos_stg3)
            info_lst.append(cu_pos)
            info_lst.append(cu_size_stg2)
            info_lst.append(cu_size_stg3)
            info_lst.append(cu_size)
            info_lst.append(split_stg2)
            info_lst.append(split_stg3)
            info_lst.append(split)
            info_lst.append(RDs)
            # Other data
            info_lst.append(orig_pos_x)
            info_lst.append(orig_pos_y)
            info_lst.append(orig_size_h)
            info_lst.append(orig_size_w)
            info_lst.append(POC)
            info_lst.append(pic_name)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class CUDatasetStg3Compl(CUDatasetBase):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again. For stage 3
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            @param files_path (string): Path to the files with annotations.
            @param channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg3Compl, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            @param entry (int): An instance from the labels.

            @return out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        color_ch = entry[8]

        if color_ch == self.channel:
            # Add Real CU
            CU_Y = entry[7]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = torch.reshape(torch.tensor(entry[2]), (-1, 2))
            cu_pos_stg2 = torch.reshape(torch.tensor(entry[1]), (-1, 2))

            # CU sizes within frame
            cu_size = torch.reshape(torch.tensor(entry[4]), (-1, 2))
            cu_size_stg2 = torch.reshape(torch.tensor(entry[3]), (-1, 2))

            # Best split for CU
            split = entry[6]
            split_stg2 = entry[5]

            # Rate distortion costs
            RDs = torch.reshape(torch.tensor(entry[0]), (-1, 6))

            # Other information
            POC = entry[9]
            pic_name = entry[10]
            orig_pos_x = entry[11][1]
            orig_pos_y = entry[11][0]
            orig_size_h = entry[12][1]
            orig_size_w = entry[12][0]

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos_stg2)
            info_lst.append(cu_pos)
            info_lst.append(cu_size_stg2)
            info_lst.append(cu_size)
            info_lst.append(split_stg2)
            info_lst.append(split)
            info_lst.append(RDs)
            # Other data
            info_lst.append(orig_pos_x)
            info_lst.append(orig_pos_y)
            info_lst.append(orig_size_h)
            info_lst.append(orig_size_w)
            info_lst.append(POC)
            info_lst.append(pic_name)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class SamplerStg6(torch.utils.data.Sampler):

    def __init__(self, data_source, batch_size):
        """!

        Args:
            @param data_source (Dataset): dataset to sample from
            @param batch_size (int): Batch size to sample data
        """
        self.data_source = data_source
        self.batch_size = batch_size 
        super(SamplerStg6, self).__init__(data_source)

        
    def __iter__(self):

        # Initialize variables
        data_size = len(self.data_source)
        indices = np.arange(data_size)
        indices = np.random.permutation(indices)
        random.shuffle(indices)
        dic_type_size = {}

        # Search for data
        for i in indices:
            # Build unique key 
            key = str(self.data_source[i][8].squeeze()[0].item()) + str(self.data_source[i][8].squeeze()[1].item()) \
                  + str(self.data_source[i][9].squeeze()[0].item()) + str(self.data_source[i][9].squeeze()[1].item()) \
                  + str(self.data_source[i][10].squeeze()[0].item()) + str(self.data_source[i][10].squeeze()[1].item())

            # Verify if key exists
            # If exists, add index to it
            # Else, create it and add index to it
            try:
                dic_type_size[key].append(i)

            except:
                dic_type_size[key] = []
                dic_type_size[key].append(i)

            # Check if for each list the size equals the batch size or note
            # if it does, yield it and reset the list
            for k in dic_type_size.keys():
                if len(dic_type_size[k]) == self.batch_size:
                    yield dic_type_size[k]
                    dic_type_size[k] = []

        # if the  loop is exited and didn't achieved batch_size
        # Yield the rest
        for k in dic_type_size.keys():
            if len(dic_type_size[k]) > 0:
                yield dic_type_size[k]

    def __len__(self):
        return self.batch_size

class SamplerStg5(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        """!

        Args:
            @param data_source (Dataset): dataset to sample from
            @param batch_size (int): Batch size to sample data
        """
        self.data_source = data_source
        self.batch_size = batch_size 
        super(SamplerStg5, self).__init__(data_source)
        
    def __iter__(self):

        # Initialize variables
        data_size = len(self.data_source)
        indices = np.arange(data_size)
        indices = np.random.permutation(indices)
        random.shuffle(indices)
        dic_type_size = {}

        # Search for data
        for i in indices:
            # Build unique key 
            key = str(self.data_source[i][7].squeeze()[0].item()) + str(self.data_source[i][7].squeeze()[1].item()) \
                  + str(self.data_source[i][8].squeeze()[0].item()) + str(self.data_source[i][8].squeeze()[1].item()) \

            # Verify if key exists
            # If exists, add index to it
            # Else, create it and add index to it
            try:
                dic_type_size[key].append(i)

            except:
                dic_type_size[key] = []
                dic_type_size[key].append(i)

            # Check if for each list the size equals the batch size or note
            # if it does, yield it and reset the list
            for k in dic_type_size.keys():
                if len(dic_type_size[k]) == self.batch_size:
                    yield dic_type_size[k]
                    dic_type_size[k] = []

        # if the  loop is exited and didn't achieved batch_size
        # Yield the rest
        for k in dic_type_size.keys():
            if len(dic_type_size[k]) > 0:
                yield dic_type_size[k]

    def __len__(self):
        return self.batch_size

class SamplerStg4(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        """!

        Args:
            @param data_source (Dataset): dataset to sample from
            @param batch_size (int): Batch size to sample data
        """
        self.data_source = data_source
        self.batch_size = batch_size 
        super(SamplerStg4, self).__init__(data_source)
        
    def __iter__(self):

        # Initialize variables
        data_size = len(self.data_source)
        indices = np.arange(data_size)
        indices = np.random.permutation(indices)
        random.shuffle(indices)
        dic_type_size = {}

        # Search for data for stage 4 and 5
        for i in indices:

            # Build unique key 
            key = str(self.data_source[i][4].squeeze()[0].item()) + str(self.data_source[i][4].squeeze()[1].item()) \
                  + str(self.data_source[i][5].squeeze()[0].item()) + str(self.data_source[i][5].squeeze()[1].item())\
                  + str(self.data_source[i][6].squeeze()[0].item()) + str(self.data_source[i][6].squeeze()[1].item())

            # Verify if key exists
            # If exists, add index to it
            # Else, create it and add index to it
            try:
                dic_type_size[key].append(i)

            except:
                dic_type_size[key] = []
                dic_type_size[key].append(i)

            # Check if for each list the size equals the batch size or note
            # if it does, yield it and reset the list
            for k in dic_type_size.keys():
                if len(dic_type_size[k]) == self.batch_size:
                    yield dic_type_size[k]
                    dic_type_size[k] = []

        # if the  loop is exited and didn't achieved batch_size
        # Yield the rest
        for k in dic_type_size.keys():
            if len(dic_type_size[k]) > 0:
                yield dic_type_size[k]

    def __len__(self):
        return self.batch_size

        
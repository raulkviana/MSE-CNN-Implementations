"""@package docstring 

@file CustomDataset.py 

@brief This library contains usefull functions to visualise data and also Classes to store and organise data structures. 
 
@section libraries_CustomDataset Libraries 
- os
- torch
- warnings
- numpy
- dataset_utils
- pandas
- cv2
- random
- __future__
- torch.utils.data
- re

@section classes_CustomDataset Classes 
- CUDatasetCU 
- CUDatasetStg 
- CUDatasetStg_v2 
- CUDatasetStg_v3 
- CUDatasetStg_v3_mod 
- CUDatasetStg_v4 
- CUDatasetStg_v4_2 
- CUDatasetStg_v4_3 
- CUDatasetStg_v5 
- CUDatasetStg_v5_eval 
- CUDatasetStg_v5_eval_32x32 
- CUDatasetStg_v5_32x32 
- CUDatasetStg_v5_16x16 
- CUDatasetStg_v5_8x8 
- CUDatasetStg_v5_8x4 
- CUDatasetStg4V5 
- CUDatasetStg5V5 
- CUDatasetStg6V5 
- CUDatasetStgAllComplexityV5 
- CUDatasetStg2ComplV5 
- CUDatasetStg6ComplV5 
- CUDatasetStg3ComplV5 
- CUDatasetStg4ComplV5 
- SamplerStg4 
- SamplerStg5 
- SamplerStg6 
- Sampler_8x8 

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


# Imports
from __future__ import print_function, division
import os
import torch
import pandas as pd
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

def yuv2bgr(matrix):
    """!
    @brief Converts yuv matrix to bgr matrix

    @param [in] matrix: Yuv matrix
    @param [out] bgr: Bgr conversion
    """

    # Convert from yuv to bgr
    bgr = cv2.cvtColor(matrix, cv2.COLOR_YUV2BGR_I420)  # cv2.COLOR_YUV2BGR_NV21)

    return bgr


def bgr2yuv(matrix):
    """!
    @brief Converts BGR matrix to YUV matrix

    @param [in] matrix: BGR matrix
    @param [out] YUV: YUV conversion
    """

    # Convert from bgr to yuv
    YUV = cv2.cvtColor(matrix, cv2.COLOR_BGR2YUV_I420)  # cv2.COLOR_YUV2BGR_NV21)

    return YUV


def get_cu_old(f_path, f_size, cu_pos, cu_size, frame_number):
    """!
    @brief Get CU from image

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
    for i in range(frame_number):
        yuv_file.read_raw()

    ret, frame_CU = yuv_file.read_raw()  # Read Frame from File

    # Return false in case a wrongfull value is returned
    if not ret:
        return False

    # Convert frame to BGR color space
    frame_CU_bgr = yuv2bgr(frame_CU)

    # Get region that contain the CU
    CU_region_BGR = frame_CU_bgr[cu_pos[0]: (cu_size[0] + cu_pos[0]), cu_pos[1]: (cu_size[1] + cu_pos[1]), :]  # CU Luma component
    # Convert region to YUV color space
    CU = bgr2yuv(CU_region_BGR)
    CU_flat = CU.flatten()

    ## Get the CU different components
    # Information about the region
    num_Ys = cu_size[1] * cu_size[0]  # Number of pixels/Luminance components
    num_UVs = int(num_Ys / 4)  # Number of each blue and red Chroma components
    # Get the components
    CU_Y = CU_flat[0: (num_Ys)]  # CU Luma component
    CU_U = CU_flat[(num_Ys): (num_Ys + num_UVs)]  # CU Chroma blue component
    CU_V = CU_flat[(num_Ys + num_UVs):]  # CU Chroma red component

    # Reshape luma components
    CU_Y = np.reshape(CU_Y, (-1, cu_size[1]))

    # Reshape chroma components
    CU_U = np.reshape(CU_U, (-1, cu_size[1]))
    CU_V = np.reshape(CU_V, (-1, cu_size[1]))

    return CU, frame_CU, CU_Y, CU_U, CU_V

def get_cu(f_path, f_size, cu_pos, cu_size, frame_number):
    """!
    @brief Get CU from image

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
    @brief Retrieves information about the YUV file info (width and height)

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


# Custom dataset for the cu oriented csv file
class CUDatasetCU(Dataset):
    """!
    CU dataset
    """

    def __init__(self, csv_file, root_dir, data_type=(128, 128)):
        """!
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.CUs_info = pd.read_csv(csv_file)
        # Updated CUs_info
        self.CUs_info = self.get_specific_CUs(self.CUs_info, data_type,
                                              0)  # Get labels with 128x128 CUs and from the luma channel
        self.root_dir = root_dir

    def get_specific_CUs(self, groundtruth, size, channel):
        """!
        @brief Recover CUs with a specific width and height from the groundtruth

        @param [in] groundtruth: Labels containing information about all CUs in the dataset
        @param [in] size: Tuple with height and width of the CUs the user wants (height, width)
        @param [in] channel: Number containing information which channel the user wants from the groundtruth (0 - Luma, 1 - Chroma, -1 - Both channels)
        @param [out] updated_gt: New version of the groundtruth updated
        """

        # Initialize matrix
        updated_gt = np.array([])
        # Flag
        done_once = False
        # Search for CUs with the specific size
        if size is not None:
            for x in range(0, groundtruth.shape[0]):

                # Get either luma or chroma channel
                if channel != -1:

                    if groundtruth.iloc[x, 5] == size[1] and groundtruth.iloc[x, 6] == size[0] and \
                            groundtruth.iloc[x, 1] == channel:

                        if not done_once:
                            updated_gt = np.array([groundtruth.iloc[x, :]])
                            done_once = not done_once  # Flip flag
                        else:
                            updated_gt = np.vstack([updated_gt, groundtruth.iloc[x, :]])  # Append row to new matrix

                # Get both luma and chroma channel
                else:

                    if groundtruth.iloc[x, 5] == size[1] and groundtruth.iloc[x, 6] == size[0]:

                        if not done_once:
                            updated_gt = np.array([groundtruth.iloc[x, :]])
                            done_once = not done_once  # Flip flag
                        else:
                            updated_gt = np.vstack([updated_gt, groundtruth.iloc[x, :]])  # Append row to new matrix
        else:
            for x in range(0, groundtruth.shape[0]):

                # Get either luma or chroma channel
                if channel != -1:

                    if groundtruth.iloc[x, 1] == channel:

                        if not done_once:
                            updated_gt = np.array([groundtruth.iloc[x, :]])
                            done_once = not done_once  # Flip flag
                        else:
                            updated_gt = np.vstack([updated_gt, groundtruth.iloc[x, :]])  # Append row to new matrix

                # Get both luma and chroma channel
                else:
                    if not done_once:
                        updated_gt = np.array([groundtruth.iloc[x, :]])
                        one_once = not done_once  # Flip flag
                    else:
                        updated_gt = np.vstack([updated_gt, groundtruth.iloc[x, :]])  # Append row to new matrix

        return updated_gt

    def get_labels(self):
        return self.CUs_info

    def __len__(self):
        return len(self.CUs_info)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ### Get info from label and process
        # Information to get the CU
        img_path = os.path.join(self.root_dir, self.CUs_info[idx, 0] + '.yuv')
        POC = self.CUs_info[idx, 2]
        pos_left = self.CUs_info[idx, 3]
        pos_top = self.CUs_info[idx, 4]
        width = self.CUs_info[idx, 5]
        height = self.CUs_info[idx, 6]
        file_name = self.CUs_info[idx, 0]
        f_size = get_file_size(file_name)  # Dict with the size of the image
        # Obtain CU
        cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']), (pos_top, pos_left),
                                                (height, width), POC)

        # Rate distortion costs
        RD0 = self.CUs_info[idx, 7]
        RD1 = self.CUs_info[idx, 8]
        RD2 = self.CUs_info[idx, 9]
        RD3 = self.CUs_info[idx, 10]
        RD4 = self.CUs_info[idx, 11]
        RD5 = self.CUs_info[idx, 12]
        # Unite
        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

        # Convert to Pytorch Tensor
        RDs = torch.from_numpy(RDs)
        CU_Y = torch.from_numpy(CU_Y)
        CU_U = torch.from_numpy(CU_U)
        CU_V = torch.from_numpy(CU_V)

        # Best split
        split = self.CUs_info[idx, 13]

        # Add dimension
        CU_Y = torch.unsqueeze(CU_Y, 0)
        CU_U = torch.unsqueeze(CU_U, 0)
        CU_V = torch.unsqueeze(CU_V, 0)

        # Convert to float
        CU_Y = CU_Y.to(dtype=torch.float32)
        CU_U = CU_U.to(dtype=torch.float32)
        CU_V = CU_V.to(dtype=torch.float32)

        # Create a instance of the dataset
        sample = {'CU_size': (height, width), 'CU_pos': (pos_top, pos_left),
                  'CU_Y': CU_Y, 'CU_U': CU_U, 'CU_V': CU_V, 'cu': cu, 'RDs': RDs,
                  'best_split': split}  # , 'frame_CU' : frame_CU}

        return sample


# Custom dataset for the stage oriented pickle file
class CUDatasetStg(Dataset):
    """!
    Dataset stage oriented
    """

    def __init__(self, file, root_dir, channel):
        """!
        Args:
            file (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.CUs_info = dataset_utils.file2lst(file)
        self.root_dir = root_dir
        self.channel = channel

    def __len__(self):
        return len(self.CUs_info)

    def fill_stg(self, idx, stg_num, img_path, f_size, POC, cu_size=None):
        """!
        Args:
            idx (int): Index of an instance from the dataset.
            stg_num (int): Number of the stage
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count
            cu_size (int): Specify the CU size to add to the list
        """
        stg = []
        for cu in self.CUs_info[idx]['stg_' + str(stg_num)]:
            if cu['color_ch'] == self.channel:
                pos_left = cu['cu_pos']['CU_loc_left']
                pos_top = cu['cu_pos']['CU_loc_top']
                width = cu['cu_size']['width']
                height = cu['cu_size']['height']

                # Verify cu_size variable
                if cu_size is None:
                    # Obtain CU
                    real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                 (pos_top, pos_left),
                                                                 (height, width), POC)
                    # Rate distortion costs
                    RD0 = cu['RD0']
                    RD1 = cu['RD1']
                    RD2 = cu['RD2']
                    RD3 = cu['RD3']
                    RD4 = cu['RD4']
                    RD5 = cu['RD5']
                    # Unite
                    RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                    # Convert to Pytorch Tensor
                    RDs = torch.from_numpy(RDs)
                    CU_Y = torch.from_numpy(CU_Y)
                    CU_U = torch.from_numpy(CU_U)
                    CU_V = torch.from_numpy(CU_V)

                    # Save new key-value pair
                    cu['RDs'] = RDs

                    # Add dimension
                    CU_Y = torch.unsqueeze(CU_Y, 0)
                    CU_U = torch.unsqueeze(CU_U, 0)
                    CU_V = torch.unsqueeze(CU_V, 0)

                    # Convert to float
                    CU_Y = CU_Y.to(dtype=torch.float32)
                    CU_U = CU_U.to(dtype=torch.float32)
                    CU_V = CU_V.to(dtype=torch.float32)

                    if cu['color_ch'] == 0:
                        cu['CU_Y'] = CU_Y
                    else:
                        cu['CU_U'] = CU_U
                        cu['CU_V'] = CU_V

                    # Append Cu to list corresponding to it's stage
                    stg.append(cu)

                elif cu_size == (width, height):
                        # Obtain CU
                        real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                     (pos_top, pos_left),
                                                                     (height, width), POC)
                        # Rate distortion costs
                        RD0 = cu['RD0']
                        RD1 = cu['RD1']
                        RD2 = cu['RD2']
                        RD3 = cu['RD3']
                        RD4 = cu['RD4']
                        RD5 = cu['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        CU_Y = torch.from_numpy(CU_Y)
                        CU_U = torch.from_numpy(CU_U)
                        CU_V = torch.from_numpy(CU_V)

                        # Save new key-value pair
                        cu['RDs'] = RDs

                        # Add dimension
                        CU_Y = torch.unsqueeze(CU_Y, 0)
                        CU_U = torch.unsqueeze(CU_U, 0)
                        CU_V = torch.unsqueeze(CU_V, 0)

                        # Convert to float
                        CU_Y = CU_Y.to(dtype=torch.float32)
                        CU_U = CU_U.to(dtype=torch.float32)
                        CU_V = CU_V.to(dtype=torch.float32)

                        if cu['color_ch'] == 0:
                            cu['CU_Y'] = CU_Y
                        else:
                            cu['CU_U'] = CU_U
                            cu['CU_V'] = CU_V

                        # Append Cu to list corresponding to it's stage
                        stg.append(cu)

        return stg

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ### Get info from label and process
        # Information to get CUs
        img_path = os.path.join(self.root_dir, self.CUs_info[idx]['pic_name'] + '.yuv')
        file_name = self.CUs_info[idx]['pic_name']
        POC = self.CUs_info[idx]['POC']
        f_size = get_file_size(file_name)  # Dict with the size of the image


        ## Fill the stages
        # Stage 1
        stg1 = self.fill_stg(idx, 1, img_path, f_size, POC)
        # Stage 2
        stg2 = self.fill_stg(idx, 2, img_path, f_size, POC)
        # Stage 3
        stg3 = self.fill_stg(idx, 3, img_path, f_size, POC)
        # Stage 4
        stg4 = self.fill_stg(idx, 4, img_path, f_size, POC)
        # Stage 5
        stg5 = self.fill_stg(idx, 5, img_path, f_size, POC)
        # Stage 6
        stg6 = self.fill_stg(idx, 6, img_path, f_size, POC)

        # Create a instance of the dataset
        sample = {'img_path': img_path, 'file_name': file_name, 'POC': POC, 'stg_1': stg1, 'stg_2': stg2, 'stg_3': stg3, 'stg_4': stg4, 'stg_5': stg5, 'stg_6': stg6}

        return sample


# Custom dataset for the stage oriented pickle file
class CUDatasetStg_v2(Dataset):
    """!
    Dataset stage oriented with capability of loading different files
    """

    def __init__(self, files_path, root_dir, channel=0, cu_type=None):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        self.files_path = files_path
        self.files = dataset_utils.get_files_from_folder(self.files_path, endswith = ".txt")
        self.lst_entries_nums = self.obtain_files_sizes(self.files)
        # Obtain amount of entries in all dataset files
        self.total_num_entries = 0
        for f in self.lst_entries_nums:
            self.total_num_entries += f
        self.root_dir = root_dir
        self.channel = channel
        self.cu_type = cu_type
        if self.cu_type is None:
            self.stg_mod = None

        else:
            self.stg_mod = self.dscv_stg(cu_type)

    def __len__(self):
        return self.total_num_entries

    def dscv_stg(self, cu_type):
        """!
        Args:
            cu_type (tuple): CU type to discover the most likely stage it belongs to
        """

        if cu_type == (128, 128):
            # Stage 1
            return 1

        elif cu_type == (64, 64):
            # Stage 2
            return 2

        elif min(cu_type) == 32:
            # Stage 3
            return 3

        elif min(cu_type) == 16:
            # Stage 4
            return 4

        elif min(cu_type) == 8:
            # Stage 5
            return 5

        elif min(cu_type) == 4:
            # Stage 6
            return 6

        else:
            raise Exception("cu_type probably is NONE!!")


    def fill_stg(self, entry, stg_num, img_path, f_size, POC, cu_size=None):
        """!
        Args:
            entry (int): An instance from the labels.
            stg_num (int): Number of the stage
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count
            cu_size (int): Specify the CU size to add to the list
        """

        # Initialize variable
        stg = []
        # Verify if the stage is empty or not
        if entry['stg_' + str(stg_num)] != []:

            for cu in entry['stg_' + str(stg_num)]:

                if cu['color_ch'] == self.channel:
                    pos_left = cu['cu_pos']['CU_loc_left']
                    pos_top = cu['cu_pos']['CU_loc_top']
                    width = cu['cu_size']['width']
                    height = cu['cu_size']['height']
                    # Verify cu_size variable
                    if cu_size is None:

                        if stg_num == 1:
                            # Obtain CU
                            real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                         (pos_top, pos_left),
                                                                         (height, width), POC)
                            # Rate distortion costs
                            RD0 = cu['RD0']
                            RD1 = cu['RD1']
                            RD2 = cu['RD2']
                            RD3 = cu['RD3']
                            RD4 = cu['RD4']
                            RD5 = cu['RD5']
                            # Unite
                            RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                            # Convert to Pytorch Tensor
                            RDs = torch.from_numpy(RDs)
                            CU_Y = torch.from_numpy(CU_Y)
                            CU_U = torch.from_numpy(CU_U)
                            CU_V = torch.from_numpy(CU_V)

                            # Save new key-value pair
                            cu['RDs'] = RDs

                            # Add dimension
                            CU_Y = torch.unsqueeze(CU_Y, 0)
                            CU_U = torch.unsqueeze(CU_U, 0)
                            CU_V = torch.unsqueeze(CU_V, 0)

                            # Convert to float
                            CU_Y = CU_Y.to(dtype=torch.float32)
                            CU_U = CU_U.to(dtype=torch.float32)
                            CU_V = CU_V.to(dtype=torch.float32)

                            if cu['color_ch'] == 0:
                                cu['CU_Y'] = CU_Y
                            else:
                                cu['CU_U'] = CU_U
                                cu['CU_V'] = CU_V

                            # Append Cu to list corresponding to it's stage
                            stg.append(cu)

                        else:
                            # Rate distortion costs
                            RD0 = cu['RD0']
                            RD1 = cu['RD1']
                            RD2 = cu['RD2']
                            RD3 = cu['RD3']
                            RD4 = cu['RD4']
                            RD5 = cu['RD5']
                            # Unite
                            RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                            # Convert to Pytorch Tensor
                            RDs = torch.from_numpy(RDs)
                            # Save new key-value pair
                            cu['RDs'] = RDs

                            # Append Cu to list corresponding to it's stage
                            stg.append(cu)

                    elif cu_size == (width, height):
                        if stg_num == 1:
                            # Obtain CU
                            real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                         (pos_top, pos_left),
                                                                         (height, width), POC)
                            # Rate distortion costs
                            RD0 = cu['RD0']
                            RD1 = cu['RD1']
                            RD2 = cu['RD2']
                            RD3 = cu['RD3']
                            RD4 = cu['RD4']
                            RD5 = cu['RD5']
                            # Unite
                            RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                            # Convert to Pytorch Tensor
                            RDs = torch.from_numpy(RDs)
                            CU_Y = torch.from_numpy(CU_Y)
                            CU_U = torch.from_numpy(CU_U)
                            CU_V = torch.from_numpy(CU_V)

                            # Save new key-value pair
                            cu['RDs'] = RDs

                            # Add dimension
                            CU_Y = torch.unsqueeze(CU_Y, 0)
                            CU_U = torch.unsqueeze(CU_U, 0)
                            CU_V = torch.unsqueeze(CU_V, 0)

                            # Convert to float
                            CU_Y = CU_Y.to(dtype=torch.float32)
                            CU_U = CU_U.to(dtype=torch.float32)
                            CU_V = CU_V.to(dtype=torch.float32)

                            if cu['color_ch'] == 0:
                                cu['CU_Y'] = CU_Y
                            else:
                                cu['CU_U'] = CU_U
                                cu['CU_V'] = CU_V

                            # Append Cu to list corresponding to it's stage
                            stg.append(cu)

                        else:
                            # Rate distortion costs
                            RD0 = cu['RD0']
                            RD1 = cu['RD1']
                            RD2 = cu['RD2']
                            RD3 = cu['RD3']
                            RD4 = cu['RD4']
                            RD5 = cu['RD5']
                            # Unite
                            RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                            # Convert to Pytorch Tensor
                            RDs = torch.from_numpy(RDs)

                            # Save new key-value pair
                            cu['RDs'] = RDs

                            # Append CU to list corresponding to it's stage
                            stg.append(cu)

        return stg

    def fill_stg_2(self, entry, stg_num, img_path, f_size, POC, cu_size=None):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count
            cu_size (int): Specify the CU size to add to the list
        """

        # Initialize variable
        cus = entry
        stg_info = cus["stg_"+str(stg_num)]

        if stg_info != []:

            if stg_info['color_ch'] == self.channel:
                pos_left = stg_info['cu_pos']['CU_loc_left']
                pos_top = stg_info['cu_pos']['CU_loc_top']
                width = stg_info['cu_size']['width']
                height = stg_info['cu_size']['height']
                # Verify cu_size variable
                if cu_size is None:

                    if stg_num == 1:
                        # Obtain CU
                        real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                     (pos_top, pos_left),
                                                                     (height, width), POC)
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        CU_Y = torch.from_numpy(CU_Y)
                        CU_U = torch.from_numpy(CU_U)
                        CU_V = torch.from_numpy(CU_V)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Add dimension
                        CU_Y = torch.unsqueeze(CU_Y, 0)
                        CU_U = torch.unsqueeze(CU_U, 0)
                        CU_V = torch.unsqueeze(CU_V, 0)

                        # Convert to float
                        CU_Y = CU_Y.to(dtype=torch.float32)
                        CU_U = CU_U.to(dtype=torch.float32)
                        CU_V = CU_V.to(dtype=torch.float32)

                        if stg_info['color_ch'] == 0:
                            stg_info['CU_Y'] = CU_Y
                        else:
                            stg_info['CU_U'] = CU_U
                            stg_info['CU_V'] = CU_V

                    else:
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                elif cu_size == (width, height):
                    if stg_num == 1:
                        # Obtain CU
                        real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                     (pos_top, pos_left),
                                                                     (height, width), POC)
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        CU_Y = torch.from_numpy(CU_Y)
                        CU_U = torch.from_numpy(CU_U)
                        CU_V = torch.from_numpy(CU_V)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Add dimension
                        CU_Y = torch.unsqueeze(CU_Y, 0)
                        CU_U = torch.unsqueeze(CU_U, 0)
                        CU_V = torch.unsqueeze(CU_V, 0)

                        # Convert to float
                        CU_Y = CU_Y.to(dtype=torch.float32)
                        CU_U = CU_U.to(dtype=torch.float32)
                        CU_V = CU_V.to(dtype=torch.float32)

                        if stg_info['color_ch'] == 0:
                            stg_info['CU_Y'] = CU_Y
                        else:
                            stg_info['CU_U'] = CU_U
                            stg_info['CU_V'] = CU_V

                    else:
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs


        return stg_info


    def obtain_files_sizes(self, files):
        """!
        Args:
            files (lst): List containing the names of files with CUs info
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
            idx (int): Index with the position to search for a specific entry
        """

        for num in range(0, len(self.lst_entries_nums)):

            # Add all the sizes until 'num'
            maxIndex = -1 # Because the arrays start at zero
            for n in range(0, num + 1):
                maxIndex += self.lst_entries_nums[n]

            if idx <= maxIndex:

                # Obtain index in the object
                if num == 0:
                    idx_obj = idx

                else:
                    idx_obj = idx - (maxIndex - self.lst_entries_nums[num] + 1)

                file_obj = dataset_utils.file2lst(self.files_path + "/" + self.files[num][0:-4])

                # Obtain entry
                entry = file_obj[idx_obj]

                return entry

        raise Exception("Entry not found!!")

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Select entry
        entry = self.select_entry(idx)

        ### Get info from label and process
        # Information to get CUs
        img_path = os.path.join(self.root_dir, entry['pic_name'] + '.yuv')
        file_name = entry['pic_name']
        POC = entry['POC']
        f_size = get_file_size(file_name)  # Dict with the size of the image

        ## Fill the stages
        # Stage 1
        if self.stg_mod == 1:
            stg1 = self.fill_stg_2(entry, 1, img_path, f_size, POC, self.cu_type)
        else:
            stg1 = self.fill_stg_2(entry, 1, img_path, f_size, POC)

        # Stage 2
        if self.stg_mod == 2:
            stg2 = self.fill_stg_2(entry, 2, img_path, f_size, POC, self.cu_type)
        else:
            stg2 = self.fill_stg_2(entry, 2, img_path, f_size, POC)

        # Stage 3
        if self.stg_mod == 3:
            stg3 = self.fill_stg_2(entry, 3, img_path, f_size, POC, self.cu_type)
        else:
            stg3 = self.fill_stg_2(entry, 3, img_path, f_size, POC)

        # Stage 4
        if self.stg_mod == 4:
            stg4 = self.fill_stg_2(entry, 4, img_path, f_size, POC, self.cu_type)
        else:
            stg4 = self.fill_stg_2(entry, 4, img_path, f_size, POC)

        # Stage 5
        if self.stg_mod == 5:
            stg5 = self.fill_stg_2(entry, 5, img_path, f_size, POC, self.cu_type)
        else:
            stg5 = self.fill_stg_2(entry, 5, img_path, f_size, POC)

        # Stage 6
        if self.stg_mod == 6:
            stg6 = self.fill_stg_2(entry, 6, img_path, f_size, POC, self.cu_type)
        else:
            stg6 = self.fill_stg_2(entry, 6, img_path, f_size, POC)

        # Create a instance of the dataset
        sample = {'stg_1': stg1, 'stg_2': stg2, 'stg_3': stg3, 'stg_4': stg4, 'stg_5': stg5, 'stg_6': stg6}

        return sample

# Custom dataset for the stage oriented pickle file
class CUDatasetStg_v3(Dataset):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    """

    def __init__(self, files_path, root_dir, channel=0, cu_type=None):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        self.files_path = files_path
        self.files = dataset_utils.get_files_from_folder(self.files_path, endswith = ".txt")
        self.lst_entries_nums = self.obtain_files_sizes(self.files)
        # Obtain amount of entries in all dataset files
        self.total_num_entries = 0
        for f in self.lst_entries_nums:
            self.total_num_entries += f
        self.root_dir = root_dir
        self.channel = channel
        self.cu_type = cu_type
        if self.cu_type is None:
            self.stg_mod = None

        else:
            self.stg_mod = self.dscv_stg(cu_type)

    def __len__(self):
        return self.total_num_entries

    def dscv_stg(self, cu_type):
        """!
        Args:
            cu_type (tuple): CU type to discover the most likely stage it belongs to
        """

        if cu_type == (128, 128):
            # Stage 1
            return 1

        elif cu_type == (64, 64):
            # Stage 2
            return 2

        elif min(cu_type) == 32:
            # Stage 3
            return 3

        elif min(cu_type) == 16:
            # Stage 4
            return 4

        elif min(cu_type) == 8:
            # Stage 5
            return 5

        elif min(cu_type) == 4:
            # Stage 6
            return 6

        else:
            raise Exception("cu_type probably is NONE!!")

    def fill_stg_2(self, entry, stg_num, img_path, f_size, POC, cu_size=None):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count
            cu_size (int): Specify the CU size to add to the list
        """

        # Initialize variable
        cus = entry
        stg_info = cus["stg_"+str(stg_num)]

        if stg_info != []:

            if stg_info['color_ch'] == self.channel:
                pos_left = stg_info['cu_pos']['CU_loc_left']
                pos_top = stg_info['cu_pos']['CU_loc_top']
                width = stg_info['cu_size']['width']
                height = stg_info['cu_size']['height']
                # Verify cu_size variable
                if cu_size is None:

                    if stg_num == 1:
                        # Obtain CU
                        real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                     (pos_top, pos_left),
                                                                     (height, width), POC)
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']

                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        CU_Y = torch.from_numpy(CU_Y)
                        CU_U = torch.from_numpy(CU_U)
                        CU_V = torch.from_numpy(CU_V)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Remove other entries
                        del stg_info["RD0"]
                        del stg_info["RD1"]
                        del stg_info["RD2"]
                        del stg_info["RD3"]
                        del stg_info["RD4"]
                        del stg_info["RD5"]

                        # Add dimension
                        CU_Y = torch.unsqueeze(CU_Y, 0)
                        CU_U = torch.unsqueeze(CU_U, 0)
                        CU_V = torch.unsqueeze(CU_V, 0)

                        # Convert to float
                        CU_Y = CU_Y.to(dtype=torch.float32)
                        CU_U = CU_U.to(dtype=torch.float32)
                        CU_V = CU_V.to(dtype=torch.float32)

                        if stg_info['color_ch'] == 0:
                            stg_info['CU_Y'] = CU_Y
                        else:
                            stg_info['CU_U'] = CU_U
                            stg_info['CU_V'] = CU_V

                    else:
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Remove other entries
                        del stg_info["RD0"]
                        del stg_info["RD1"]
                        del stg_info["RD2"]
                        del stg_info["RD3"]
                        del stg_info["RD4"]
                        del stg_info["RD5"]

                elif cu_size == (height, width):
                    if stg_num == 1:
                        # Obtain CU
                        real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                     (pos_top, pos_left),
                                                                     (height, width), POC)
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        CU_Y = torch.from_numpy(CU_Y)
                        CU_U = torch.from_numpy(CU_U)
                        CU_V = torch.from_numpy(CU_V)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Remove other entries
                        del stg_info["RD0"]
                        del stg_info["RD1"]
                        del stg_info["RD2"]
                        del stg_info["RD3"]
                        del stg_info["RD4"]
                        del stg_info["RD5"]

                        # Add dimension
                        CU_Y = torch.unsqueeze(CU_Y, 0)
                        CU_U = torch.unsqueeze(CU_U, 0)
                        CU_V = torch.unsqueeze(CU_V, 0)

                        # Convert to float
                        CU_Y = CU_Y.to(dtype=torch.float32)
                        CU_U = CU_U.to(dtype=torch.float32)
                        CU_V = CU_V.to(dtype=torch.float32)

                        if stg_info['color_ch'] == 0:
                            stg_info['CU_Y'] = CU_Y
                        else:
                            stg_info['CU_U'] = CU_U
                            stg_info['CU_V'] = CU_V

                    else:
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Remove other entries
                        del stg_info["RD0"]
                        del stg_info["RD1"]
                        del stg_info["RD2"]
                        del stg_info["RD3"]
                        del stg_info["RD4"]
                        del stg_info["RD5"]

                else:
                    return False  # Return false in case the cu doesnt has the correct dimensions

            else:

                raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return stg_info

    def obtain_files_sizes(self, files):
        """!
        Args:
            files (lst): List containing the names of files with CUs info
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
            idx (int): Index with the position to search for a specific entry
        """

        for num in range(0, len(self.lst_entries_nums)):

            # Add all the sizes until 'num'
            maxIndex = -1 # Because the arrays start at zero
            for n in range(0, num + 1):
                maxIndex += self.lst_entries_nums[n]

            if idx <= maxIndex:

                # Obtain index in the object
                if num == 0:
                    idx_obj = idx

                else:
                    idx_obj = idx - (maxIndex - self.lst_entries_nums[num] + 1)

                file_obj = dataset_utils.file2lst(self.files_path + "/" + self.files[num][0:-4])

                # Obtain entry
                entry = file_obj[idx_obj]

                return entry

        raise Exception("Entry not found!!")

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ok = False
        while not ok:

            # Select entry
            entry = self.select_entry(idx)

            ### Get info from label and process
            # Information to get CUs
            img_path = os.path.join(self.root_dir, entry['pic_name'] + '.yuv')
            file_name = entry['pic_name']
            POC = entry['POC']
            f_size = get_file_size(file_name)  # Dict with the size of the image

            ## Fill the stages
            # Stage 1
            if self.stg_mod == 1:
                stg1 = self.fill_stg_2(entry, 1, img_path, f_size, POC, self.cu_type)
            else:
                stg1 = self.fill_stg_2(entry, 1, img_path, f_size, POC)

            # Stage 2
            if self.stg_mod == 2:
                stg2 = self.fill_stg_2(entry, 2, img_path, f_size, POC, self.cu_type)
            else:
                stg2 = self.fill_stg_2(entry, 2, img_path, f_size, POC)

            # Stage 3
            if self.stg_mod == 3:
                stg3 = self.fill_stg_2(entry, 3, img_path, f_size, POC, self.cu_type)
            else:
                stg3 = self.fill_stg_2(entry, 3, img_path, f_size, POC)

            # Stage 4
            if self.stg_mod == 4:
                stg4 = self.fill_stg_2(entry, 4, img_path, f_size, POC, self.cu_type)
            else:
                stg4 = self.fill_stg_2(entry, 4, img_path, f_size, POC)

            # Stage 5
            if self.stg_mod == 5:
                stg5 = self.fill_stg_2(entry, 5, img_path, f_size, POC, self.cu_type)
            else:
                stg5 = self.fill_stg_2(entry, 5, img_path, f_size, POC)

            # Stage 6
            if self.stg_mod == 6:
                stg6 = self.fill_stg_2(entry, 6, img_path, f_size, POC, self.cu_type)
            else:
                stg6 = self.fill_stg_2(entry, 6, img_path, f_size, POC)

            # Verify if the data has the right data type
            if type(stg6) == bool or type(stg5) == bool or type(stg4) == bool or type(stg3) == bool or type(stg2) == bool or type(stg1) == bool:
                idx = (idx + 1) % self.total_num_entries  # Search next entry

            else:
                ok = True  # Reset flag

        # Create a instance of the dataset
        sample = {'stg_1': stg1, 'stg_2': stg2, 'stg_3': stg3, 'stg_4': stg4, 'stg_5': stg5, 'stg_6': stg6}

        return sample

# Custom dataset for the stage oriented pickle file -- used for solve imbalance datasets (It tries to find always different classes)
class CUDatasetStg_v3_mod(Dataset):
    """!
    Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    """

    def __init__(self, files_path, root_dir, channel=0, cu_type=None):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        self.files_path = files_path
        self.files = dataset_utils.get_files_from_folder(self.files_path, endswith = ".txt")
        self.lst_entries_nums = self.obtain_files_sizes(self.files)
        # Obtain amount of entries in all dataset files
        self.total_num_entries = 0
        for f in self.lst_entries_nums:
            self.total_num_entries += f
        self.root_dir = root_dir
        self.channel = channel
        self.cu_type = cu_type
        if self.cu_type is None:
            self.stg_mod = None

        else:
            self.stg_mod = self.dscv_stg(cu_type)

        self.cu_counter = 0

    def __len__(self):
        return self.total_num_entries

    def dscv_stg(self, cu_type):
        """!
        Args:
            cu_type (tuple): CU type to discover the most likely stage it belongs to
        """

        if cu_type == (128, 128):
            # Stage 1
            return 1

        elif cu_type == (64, 64):
            # Stage 2
            return 2

        elif min(cu_type) == 32:
            # Stage 3
            return 3

        elif min(cu_type) == 16:
            # Stage 4
            return 4

        elif min(cu_type) == 8:
            # Stage 5
            return 5

        elif min(cu_type) == 4:
            # Stage 6
            return 6

        else:
            raise Exception("cu_type probably is NONE!!")

    def fill_stg_2(self, entry, stg_num, img_path, f_size, POC, cu_size=None):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count
            cu_size (int): Specify the CU size to add to the list
        """

        # Initialize variable
        cus = entry
        stg_info = cus["stg_"+str(stg_num)]

        if stg_info != []:

            if stg_info['color_ch'] == self.channel:
                pos_left = stg_info['cu_pos']['CU_loc_left']
                pos_top = stg_info['cu_pos']['CU_loc_top']
                width = stg_info['cu_size']['width']
                height = stg_info['cu_size']['height']
                # Verify cu_size variable
                if cu_size is None:

                    if stg_num == 1:
                        # Obtain CU
                        real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                     (pos_top, pos_left),
                                                                     (height, width), POC)
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']

                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        CU_Y = torch.from_numpy(CU_Y)
                        CU_U = torch.from_numpy(CU_U)
                        CU_V = torch.from_numpy(CU_V)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Remove other entries
                        del stg_info["RD0"]
                        del stg_info["RD1"]
                        del stg_info["RD2"]
                        del stg_info["RD3"]
                        del stg_info["RD4"]
                        del stg_info["RD5"]

                        # Add dimension
                        CU_Y = torch.unsqueeze(CU_Y, 0)
                        CU_U = torch.unsqueeze(CU_U, 0)
                        CU_V = torch.unsqueeze(CU_V, 0)

                        # Convert to float
                        CU_Y = CU_Y.to(dtype=torch.float32)
                        CU_U = CU_U.to(dtype=torch.float32)
                        CU_V = CU_V.to(dtype=torch.float32)

                        if stg_info['color_ch'] == 0:
                            stg_info['CU_Y'] = CU_Y
                        else:
                            stg_info['CU_U'] = CU_U
                            stg_info['CU_V'] = CU_V

                    else:
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Remove other entries
                        del stg_info["RD0"]
                        del stg_info["RD1"]
                        del stg_info["RD2"]
                        del stg_info["RD3"]
                        del stg_info["RD4"]
                        del stg_info["RD5"]

                elif cu_size == (width, height):
                    if stg_num == 1:
                        # Obtain CU
                        real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                                     (pos_top, pos_left),
                                                                     (height, width), POC)
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)
                        CU_Y = torch.from_numpy(CU_Y)
                        CU_U = torch.from_numpy(CU_U)
                        CU_V = torch.from_numpy(CU_V)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Remove other entries
                        del stg_info["RD0"]
                        del stg_info["RD1"]
                        del stg_info["RD2"]
                        del stg_info["RD3"]
                        del stg_info["RD4"]
                        del stg_info["RD5"]

                        # Add dimension
                        CU_Y = torch.unsqueeze(CU_Y, 0)
                        CU_U = torch.unsqueeze(CU_U, 0)
                        CU_V = torch.unsqueeze(CU_V, 0)

                        # Convert to float
                        CU_Y = CU_Y.to(dtype=torch.float32)
                        CU_U = CU_U.to(dtype=torch.float32)
                        CU_V = CU_V.to(dtype=torch.float32)

                        if stg_info['color_ch'] == 0:
                            stg_info['CU_Y'] = CU_Y
                        else:
                            stg_info['CU_U'] = CU_U
                            stg_info['CU_V'] = CU_V

                    else:
                        # Rate distortion costs
                        RD0 = stg_info['RD0']
                        RD1 = stg_info['RD1']
                        RD2 = stg_info['RD2']
                        RD3 = stg_info['RD3']
                        RD4 = stg_info['RD4']
                        RD5 = stg_info['RD5']
                        # Unite
                        RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                        # Convert to Pytorch Tensor
                        RDs = torch.from_numpy(RDs)

                        # Save new key-value pair
                        stg_info['RDs'] = RDs

                        # Remove other entries
                        del stg_info["RD0"]
                        del stg_info["RD1"]
                        del stg_info["RD2"]
                        del stg_info["RD3"]
                        del stg_info["RD4"]
                        del stg_info["RD5"]

            else:

                raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return stg_info


    def obtain_files_sizes(self, files):
        """!
        Args:
            files (lst): List containing the names of files with CUs info
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
            idx (int): Index with the position to search for a specific entry
        """

        for num in range(0, len(self.lst_entries_nums)):

            # Add all the sizes until 'num'
            maxIndex = -1 # Because the arrays start at zero
            for n in range(0, num + 1):
                maxIndex += self.lst_entries_nums[n]

            if idx <= maxIndex:

                # Obtain index in the object
                if num == 0:
                    idx_obj = idx

                else:
                    idx_obj = idx - (maxIndex - self.lst_entries_nums[num] + 1)

                file_obj = dataset_utils.file2lst(self.files_path + "/" + self.files[num][0:-4])

                # Obtain entry
                entry = file_obj[idx_obj]

                return entry

        raise Exception("Entry not found!!")

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ok = False
        while not ok:

            # Select entry
            entry = self.select_entry(idx)

            ### Get info from label and process
            # Information to get CUs
            img_path = os.path.join(self.root_dir, entry['pic_name'] + '.yuv')
            file_name = entry['pic_name']
            POC = entry['POC']
            f_size = get_file_size(file_name)  # Dict with the size of the image

            if self.cu_counter % 2 == 0 and entry["stg_2"]["split"] == 1: # It has to be a non-split
                idx = (idx + 1) % (self.total_num_entries)
                continue

            ok = True
            self.cu_counter += 1

            ## Fill the stages
            # Stage 1
            if self.stg_mod == 1:
                stg1 = self.fill_stg_2(entry, 1, img_path, f_size, POC, self.cu_type)
            else:
                stg1 = self.fill_stg_2(entry, 1, img_path, f_size, POC)

            # Stage 2
            if self.stg_mod == 2:
                stg2 = self.fill_stg_2(entry, 2, img_path, f_size, POC, self.cu_type)
            else:
                stg2 = self.fill_stg_2(entry, 2, img_path, f_size, POC)

            # Stage 3
            if self.stg_mod == 3:
                stg3 = self.fill_stg_2(entry, 3, img_path, f_size, POC, self.cu_type)
            else:
                stg3 = self.fill_stg_2(entry, 3, img_path, f_size, POC)

            # Stage 4
            if self.stg_mod == 4:
                stg4 = self.fill_stg_2(entry, 4, img_path, f_size, POC, self.cu_type)
            else:
                stg4 = self.fill_stg_2(entry, 4, img_path, f_size, POC)

            # Stage 5
            if self.stg_mod == 5:
                stg5 = self.fill_stg_2(entry, 5, img_path, f_size, POC, self.cu_type)
            else:
                stg5 = self.fill_stg_2(entry, 5, img_path, f_size, POC)

            # Stage 6
            if self.stg_mod == 6:
                stg6 = self.fill_stg_2(entry, 6, img_path, f_size, POC, self.cu_type)
            else:
                stg6 = self.fill_stg_2(entry, 6, img_path, f_size, POC)

        # Create a instance of the dataset
        sample = {'stg_1': stg1, 'stg_2': stg2, 'stg_3': stg3, 'stg_4': stg4, 'stg_5': stg5, 'stg_6': stg6}

        return sample

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v4(Dataset):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - Multi Batch
    """

    def __init__(self, files_path, root_dir, stg_num, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        self.files_path = files_path
        self.files = dataset_utils.get_files_from_folder(self.files_path, endswith = ".txt")
        self.lst_entries_nums = self.obtain_files_sizes(self.files)
        # Obtain amount of entries in all dataset files
        self.total_num_entries = 0
        for f in self.lst_entries_nums:
            self.total_num_entries += f
        self.root_dir = root_dir
        self.channel = channel
        self.stg_num = stg_num

    def __len__(self):
        return self.total_num_entries

    def get_sample(self, entry, stg_num, img_path, f_size, POC):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        cus = entry
        info_lst = []
        stg_info = cus["stg_" + str(stg_num)]
        stg_info_1 = cus["stg_1"]

        if stg_info != []:

            if stg_info['color_ch'] == self.channel:
                pos_left = stg_info_1['cu_pos']['CU_loc_left']
                pos_top = stg_info_1['cu_pos']['CU_loc_top']
                width = stg_info_1['cu_size']['width']
                height = stg_info_1['cu_size']['height']

                # Obtain CU
                real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                             (pos_top, pos_left),
                                                             (height, width), POC)

                # Rate distortion costs
                RD0 = stg_info['RD0']
                RD1 = stg_info['RD1']
                RD2 = stg_info['RD2']
                RD3 = stg_info['RD3']
                RD4 = stg_info['RD4']
                RD5 = stg_info['RD5']

                # Unite
                RDs = np.reshape(np.array([[RD0, RD1, RD2, RD3, RD4, RD5]]), (-1, 6))

                # Convert to Pytorch Tensor
                RDs = torch.from_numpy(RDs)
                CU_Y = torch.from_numpy(CU_Y)
                CU_U = torch.from_numpy(CU_U)
                CU_V = torch.from_numpy(CU_V)

                # Add dimension
                CU_Y = torch.unsqueeze(CU_Y, 0)
                CU_U = torch.unsqueeze(CU_U, 0)
                CU_V = torch.unsqueeze(CU_V, 0)

                # Convert to float
                CU_Y = CU_Y.to(dtype=torch.float32)
                CU_U = CU_U.to(dtype=torch.float32)
                CU_V = CU_V.to(dtype=torch.float32)

                # Save values
                if stg_info['color_ch'] == 0:
                    info_lst.append(CU_Y)
                else:
                    info_lst.append(CU_U)
                    info_lst.append(CU_V)

                info_lst.append(RDs)
                info_lst.append(stg_info_1["cu_pos"]["CU_loc_left"])
                info_lst.append(stg_info_1["cu_pos"]["CU_loc_top"])
                info_lst.append(stg_info["cu_pos"]["CU_loc_left"])
                info_lst.append(stg_info["cu_pos"]["CU_loc_top"])
                info_lst.append(stg_info["split"])

            else:
                raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        else:
            #raise Exception("This can not happen! The stage is empty...")
            pass

        return info_lst

    def obtain_files_sizes(self, files):
        """!
        Args:
            files (lst): List containing the names of files with CUs info
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
            idx (int): Index with the position to search for a specific entry
        """

        for num in range(0, len(self.lst_entries_nums)):

            # Add all the sizes until 'num'
            maxIndex = -1 # Because the arrays start at zero
            for n in range(0, num + 1):
                maxIndex += self.lst_entries_nums[n]

            if idx <= maxIndex:

                # Obtain index in the object
                if num == 0:
                    idx_obj = idx

                else:
                    idx_obj = idx - (maxIndex - self.lst_entries_nums[num] + 1)

                file_obj = dataset_utils.file2lst(self.files_path + "/" + self.files[num][0:-4])

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
            # Information to get CUs
            img_path = os.path.join(self.root_dir, entry['pic_name'] + '.yuv')
            file_name = entry['pic_name']
            POC = entry['POC']
            f_size = get_file_size(file_name)  # Dict with the size of the image

            # Get sample
            sample = self.get_sample(entry, self.stg_num, img_path, f_size, POC)

            ctu_size = (sample[0].shape[-2], sample[0].shape[-1])
            if entry["stg_" + str(self.stg_num)] != [] and (128, 128) == ctu_size:
                found = True

            else:
                idx = (idx + 1) % self.total_num_entries  # Increment index until a acceptable entry is found

        return sample


# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v4_2(Dataset):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - Multi Batch
    """

    def __init__(self, files_path, root_dir, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        self.files_path = files_path
        self.files = dataset_utils.get_files_from_folder(self.files_path, endswith = ".txt")
        self.lst_entries_nums = self.obtain_files_sizes(self.files)
        # Obtain amount of entries in all dataset files
        self.total_num_entries = 0
        for f in self.lst_entries_nums:
            self.total_num_entries += f
        self.root_dir = root_dir
        self.channel = channel

    def __len__(self):
        return self.total_num_entries

    def get_sample(self, entry, img_path, f_size, POC):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        ctu = entry["CTU"]

        if ctu['color_ch'] == self.channel:
            pos_left = ctu['cu_pos']['CU_loc_left']
            pos_top = ctu['cu_pos']['CU_loc_top']
            width = ctu['cu_size']['width']
            height = ctu['cu_size']['height']

            # Obtain CU
            real_cu, frame_CU, CU_Y, CU_U, CU_V = get_cu(img_path, (f_size['height'], f_size['width']),
                                                         (pos_top, pos_left),
                                                         (height, width), POC)

            # Rate distortion costs
            RDs = entry['RD']

            # Unite
            RDs = np.reshape(RDs, (-1, 6))

            # Convert to Pytorch Tensor
            CU_Y = torch.from_numpy(CU_Y)
            CU_U = torch.from_numpy(CU_U)
            CU_V = torch.from_numpy(CU_V)

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)
            CU_U = torch.unsqueeze(CU_U, 0)
            CU_V = torch.unsqueeze(CU_V, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)
            CU_U = CU_U.to(dtype=torch.float32)
            CU_V = CU_V.to(dtype=torch.float32)

            # Save values
            if ctu['color_ch'] == 0:
                info_lst.append(CU_Y)
            else:
                info_lst.append(CU_U)
                info_lst.append(CU_V)

            info_lst.append(RDs)
            info_lst.append(entry["best_splits"])
            info_lst.append(entry["positions"])  # Positions
            info_lst.append(torch.tensor([ctu['cu_pos']["CU_loc_top"], ctu['cu_pos']["CU_loc_left"]]))  # Delete later

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

    def obtain_files_sizes(self, files):
        """!
        Args:
            files (lst): List containing the names of files with CUs info
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
            idx (int): Index with the position to search for a specific entry
        """

        for num in range(0, len(self.lst_entries_nums)):

            # Add all the sizes until 'num'
            maxIndex = -1 # Because the arrays start at zero
            for n in range(0, num + 1):
                maxIndex += self.lst_entries_nums[n]

            if idx <= maxIndex:

                # Obtain index in the object
                if num == 0:
                    idx_obj = idx

                else:
                    idx_obj = idx - (maxIndex - self.lst_entries_nums[num] + 1)

                file_obj = dataset_utils.file2lst(self.files_path + "/" + self.files[num][0:-4])

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
            # Information to get CUs
            img_path = os.path.join(self.root_dir, entry['pic_name'] + '.yuv')
            file_name = entry['pic_name']
            POC = entry['POC']
            f_size = get_file_size(file_name)  # Dict with the size of the image

            # Get sample
            sample = self.get_sample(entry, img_path, f_size, POC)

            ctu_size = (sample[0].shape[-2], sample[0].shape[-1])
            if (128, 128) == ctu_size:
                found = True

            else:
                idx = (idx + 1) % self.total_num_entries  # Increment index until a acceptable entry is found

        return sample

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v4_3(Dataset):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - Multi Batch
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        self.files_path = files_path
        self.files = dataset_utils.get_files_from_folder(self.files_path, endswith = ".txt")
        self.lst_entries_nums = self.obtain_files_sizes(self.files)
        # Obtain amount of entries in all dataset files
        self.total_num_entries = 0
        for f in self.lst_entries_nums:
            self.total_num_entries += f
        self.channel = channel
        self.index_lims = []
        for k in range(len(self.lst_entries_nums)):
            sum = -1

            for f in self.lst_entries_nums[:k+1]:
                sum += f

            self.index_lims.append(sum)

        self.data_files = []
        for k in range(len(self.files)):
            self.data_files.append(dataset_utils.file2lst(self.files_path + "/" + self.files[k][0:-4]))

    def __len__(self):
        return self.total_num_entries

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        ctu = entry["CTU"]

        if ctu['color_ch'] == self.channel:
            # Rate distortion costs
            RDs = entry['RD']

            # Add Luma
            CU_Y = entry["real_CTU"]

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(RDs)
            info_lst.append(entry["best_splits"])

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

    def obtain_files_sizes(self, files):
        """!
        Args:
            files (lst): List containing the names of files with CUs info
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
            idx (int): Index with the position to search for a specific entry
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

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v5(Dataset):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """

        self.files_path = files_path
        self.files = dataset_utils.get_files_from_folder(self.files_path, endswith = ".txt")
        self.lst_entries_nums = self.obtain_files_sizes(self.files)
        # Obtain amount of entries in all dataset files
        self.total_num_entries = 0
        for f in self.lst_entries_nums:
            self.total_num_entries += f
        self.channel = channel
        self.index_lims = []
        for k in range(len(self.lst_entries_nums)):
            sum = -1

            for f in self.lst_entries_nums[:k+1]:
                sum += f

            self.index_lims.append(sum)

        self.data_files = []
        for k in range(len(self.files)):
            self.data_files.append(dataset_utils.file2lst(self.files_path + "/" + self.files[k][0:-4]))

    def __len__(self):
        return self.total_num_entries

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        # try:

        # Add Luma
        CU_Y = entry["real_CTU"]
        #print("Shape inside getsample before unqueeze:",CU_Y.shape)

        # Convert to Pytorch Tensor
        #CU_Y = torch.from_numpy(CU_Y)

        # Add dimension
        CU_Y = torch.unsqueeze(CU_Y, 0)
        #print("Shape inside getsample after unqueeze:",CU_Y.shape)

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
            files (lst): List containing the names of files with CUs info
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
            idx (int): Index with the position to search for a specific entry
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

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v5_eval(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 2 evaluation
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        super(CUDatasetStg_v5_eval, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        ctu = entry["CTU"]

        if ctu['color_ch'] == self.channel:
            # Add Luma
            CU_Y = entry["real_CTU"]

            CU_Y = torch.unsqueeze(CU_Y, 0)
            #print("Shape inside getsample after unqueeze:",CU_Y.shape)

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
            
            # Other information about the CU
            orig_pos_x = entry["orig_pos_x"]
            orig_pos_y = entry["orig_pos_y"]
            orig_size_h = entry["orig_size_h"]
            orig_size_w = entry["orig_size_w"]
            POC = entry["POC"]
            pic_name = entry["pic_name"]

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos)
            info_lst.append(cu_size)
            info_lst.append(split)
            info_lst.append(RDs)
            # Save other infos
            info_lst.append(orig_pos_x)
            info_lst.append(orig_pos_y)
            info_lst.append(orig_size_h)
            info_lst.append(orig_size_w)
            info_lst.append(POC)
            info_lst.append(pic_name)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst

class CUDatasetStg_v5_eval_32x32(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 3 evaluation
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        super(CUDatasetStg_v5_eval_32x32, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        ctu = entry["CTU"]

        if ctu['color_ch'] == self.channel:
            # Add Luma
            CU_Y = entry["real_CTU"]

            CU_Y = torch.unsqueeze(CU_Y, 0)
            #print("Shape inside getsample after unqueeze:",CU_Y.shape)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = entry["cu_pos_stg3"]
            cu_pos_stg2 = entry["cu_pos_stg2"]

            # CU sizes within frame
            cu_size = entry["cu_size_stg3"]
            cu_size_stg2 = entry["cu_size_stg2"]

            # Best split for CU
            split = entry["split"]
            split_stg2 = entry["split_stg2"]

            # Rate distortion costs
            RDs = entry['RD']

            # Unite
            RDs = np.reshape(RDs, (-1, 6))

            # Other information about the CU
            orig_pos_x = entry["orig_pos_x"]
            orig_pos_y = entry["orig_pos_y"]
            orig_size_h = entry["orig_size_h"]
            orig_size_w = entry["orig_size_w"]
            POC = entry["POC"]
            pic_name = entry["pic_name"]

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

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v5_32x32(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 3
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        super(CUDatasetStg_v5_32x32, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        ctu = entry["CTU"]

        if ctu['color_ch'] == self.channel:
            # Add Luma
            CU_Y = entry["real_CTU"]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = entry["cu_pos_stg3"]
            cu_pos_stg2 = entry["cu_pos_stg2"]

            # CU sizes within frame
            cu_size = entry["cu_size_stg3"]
            cu_size_stg2 = entry["cu_size_stg2"]

            # Best split for CU
            split = entry["split"]
            split_stg2 = entry["split_stg2"]

            # Rate distortion costs
            RDs = entry['RD']

            # Unite
            RDs = np.reshape(RDs, (-1, 6))

            # Save values
            info_lst.append(CU_Y)
            info_lst.append(cu_pos_stg2)
            info_lst.append(cu_pos)
            info_lst.append(cu_size_stg2)
            info_lst.append(cu_size)
            info_lst.append(split_stg2)
            info_lst.append(split)
            info_lst.append(RDs)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst


# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v5_16x16(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 4
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
            cu_type: Type of CU to filter dataset (64x64, 32x32, 64x32)
        """
        super(CUDatasetStg_v5_16x16, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        ctu = entry["CTU"]

        if ctu['color_ch'] == self.channel:
            # Add Luma
            CU_Y = entry["real_CTU"]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = entry["cu_pos_stg4"]
            cu_pos_stg3 = entry["cu_pos_stg3"]
            cu_pos_stg2 = entry["cu_pos_stg2"]

            # CU sizes within frame
            cu_size = entry["cu_size_stg4"]
            cu_size_stg3 = entry["cu_size_stg3"]
            cu_size_stg2 = entry["cu_size_stg2"]

            # Best split for CU
            split = entry["split"]
            split_stg3 = entry["split_stg3"]
            split_stg2 = entry["split_stg2"]

            # Rate distortion costs
            RDs = entry['RD']

            # Unite
            RDs = np.reshape(RDs, (-1, 6))

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


# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v5_8x8(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 5
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg_v5_8x8, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        ctu = entry["CTU"]

        if ctu['color_ch'] == self.channel:
            # Add Luma
            CU_Y = entry["real_CTU"]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = entry["cu_pos_stg5"]
            cu_pos_stg4 = entry["cu_pos_stg4"]
            cu_pos_stg3 = entry["cu_pos_stg3"]
            cu_pos_stg2 = entry["cu_pos_stg2"]

            # CU sizes within frame
            cu_size = entry["cu_size_stg5"]
            cu_size_stg4 = entry["cu_size_stg4"]
            cu_size_stg3 = entry["cu_size_stg3"]
            cu_size_stg2 = entry["cu_size_stg2"]

            # Best split for CU
            split = entry["split"]
            split_stg4 = entry["split_stg4"]
            split_stg3 = entry["split_stg3"]
            split_stg2 = entry["split_stg2"]

            # Rate distortion costs
            RDs = entry['RD']

            # Unite
            RDs = np.reshape(RDs, (-1, 6))

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


# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg_v5_8x4(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 5
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg_v5_8x4, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
        """

        # Initialize variable
        info_lst = []
        ctu = entry["CTU"]

        if ctu['color_ch'] == self.channel:
            # Add Luma
            CU_Y = entry["real_CTU"]

            # Add dimension
            CU_Y = torch.unsqueeze(CU_Y, 0)

            # Convert to float
            CU_Y = CU_Y.to(dtype=torch.float32)

            # CU positions within frame
            cu_pos = entry["cu_pos_stg6"]
            cu_pos_stg5 = entry["cu_pos_stg5"]
            cu_pos_stg4 = entry["cu_pos_stg4"]
            cu_pos_stg3 = entry["cu_pos_stg3"]
            cu_pos_stg2 = entry["cu_pos_stg2"]

            # CU sizes within frame
            cu_size = entry["cu_size_stg6"]
            cu_size_stg5 = entry["cu_size_stg5"]
            cu_size_stg4 = entry["cu_size_stg4"]
            cu_size_stg3 = entry["cu_size_stg3"]
            cu_size_stg2 = entry["cu_size_stg2"]

            # Best split for CU
            split = entry["split"]
            split_stg5 = entry["split_stg5"]
            split_stg4 = entry["split_stg4"]
            split_stg3 = entry["split_stg3"]
            split_stg2 = entry["split_stg2"]

            # Rate distortion costs
            RDs = entry['RD']

            # Unite
            RDs = np.reshape(RDs, (-1, 6))

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


# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg4V5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 4
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg4V5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
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



# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg6V5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 6
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg6V5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.

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

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStgAllComplexityV5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage all
    - Evaluate complexity
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStgAllComplexityV5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.

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
            orig_pos_x_stg1 = entry[20][1]
            orig_pos_y_stg1 = entry[20][0]
            orig_size_h_stg1 = entry[21][1]
            orig_size_w_stg1 = entry[21][0]
            orig_pos_x_stg2 = entry[22][1]
            orig_pos_y_stg2 = entry[22][0]
            orig_size_h_stg2 = entry[23][1]
            orig_size_w_stg2 = entry[23][0]
            orig_pos_x_stg3 = entry[24][1]
            orig_pos_y_stg3 = entry[24][0]
            orig_size_h_stg3 = entry[25][1]
            orig_size_w_stg3 = entry[25][0]
            orig_pos_x_stg4 = entry[26][1]
            orig_pos_y_stg4 = entry[26][0]
            orig_size_h_stg4 = entry[27][1]
            orig_size_w_stg4 = entry[27][0]
            orig_pos_x_stg5 = entry[28][1]
            orig_pos_y_stg5 = entry[28][0]
            orig_size_h_stg5 = entry[29][1]
            orig_size_w_stg5 = entry[29][0]
            orig_pos_x_stg6 = entry[30][1]
            orig_pos_y_stg6 = entry[30][0]
            orig_size_h_stg6 = entry[31][1]
            orig_size_w_stg6 = entry[31][0]


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
            info_lst.append(orig_pos_x_stg1)
            info_lst.append(orig_pos_y_stg1)
            info_lst.append(orig_size_h_stg1)
            info_lst.append(orig_size_w_stg1)
            info_lst.append(orig_pos_x_stg2)
            info_lst.append(orig_pos_y_stg2)
            info_lst.append(orig_size_h_stg2)
            info_lst.append(orig_size_w_stg2)
            info_lst.append(orig_pos_x_stg3)
            info_lst.append(orig_pos_y_stg3)
            info_lst.append(orig_size_h_stg3)
            info_lst.append(orig_size_w_stg3)
            info_lst.append(orig_pos_x_stg4)
            info_lst.append(orig_pos_y_stg4)
            info_lst.append(orig_size_h_stg4)
            info_lst.append(orig_size_w_stg4)
            info_lst.append(orig_pos_x_stg5)
            info_lst.append(orig_pos_y_stg5)
            info_lst.append(orig_size_h_stg5)
            info_lst.append(orig_size_w_stg5)
            info_lst.append(orig_pos_x_stg6)
            info_lst.append(orig_pos_y_stg6)
            info_lst.append(orig_size_h_stg6)
            info_lst.append(orig_size_w_stg6)
            info_lst.append(POC)
            info_lst.append(pic_name)

        else:
            raise Exception("This can not happen! This CU color channel should be " + str(self.channel) + "Try generating the labels and obtain just a specific color channel (see dataset_utils)!")

        return info_lst



# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg5V5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 5
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg5V5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
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

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg5ComplV5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 5
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg5ComplV5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
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



# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg2ComplV5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 2
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg2ComplV5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
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

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg6ComplV5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 6
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg6ComplV5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.

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

# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg4ComplV5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 4
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg4ComplV5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
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


# Custom dataset for the stage oriented pickle file and it's multi batch
class CUDatasetStg3ComplV5(CUDatasetStg_v5):
    """!
    - Dataset stage oriented with capability of loading different files and it's supposed to be used with the function dataset_utils.change_labels_function_again
    - JF recommendations
    - For stage 4
    """

    def __init__(self, files_path, channel=0):
        """!
        Args:
            files_path (string): Path to the files with annotations.
            root_dir (string): Directory with all the images.
            channel: Channel to get for the dataset (0:Luma, 1:Chroma)
        """
        super(CUDatasetStg3ComplV5, self).__init__(files_path, channel)

    def get_sample(self, entry):
        """!
        Args:
            entry (int): An instance from the labels.
            img_path (string): Yuv image path
            f_size (dict): Size of the frame
            POC (int): Picture order count

            out: lst -  CTU | RD_for_specific_stage | cu_left_of_stg_1 | cu_top_of_stg_1 | cu_left_for_specific_stage | cu_top_for_specific_stage |  split_for_specific_stage
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


# Batch sampler

class Sampler_8x8(torch.utils.data.Sampler):
    """! Summary

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size 
        super(Sampler_8x8, self).__init__(data_source)
        
    def __iter__(self):

        # Initialize variables
        data_size = len(self.data_source)

        size_type_16x16 = []
        size_type_32x16 = []
        size_type_32x8 = []
        size_type_16x32 = []
        size_type_8x32 = []

        # Search for data with 16x16 in stage 4
        for i in range(data_size):
            # Aggregate batch_size of 16x16
            if self.data_source[i][7].squeeze()[0].item() == 16 and self.data_source[i][7].squeeze()[1].item() == 16:
                size_type_16x16.append(i)
            # yield them when you achieve batch_size
            if len(size_type_16x16) == self.batch_size:
                random.shuffle(size_type_16x16)
                yield size_type_16x16
                size_type_16x16 = []

            # Aggregate batch_size of 32x16
            if (self.data_source[i][7].squeeze()[0].item() == 32 and self.data_source[i][7].squeeze()[1].item() == 16):                size_type_32x16.append(i)
            # yield them when you achieve batch_size
            if len(size_type_32x16) == self.batch_size:
                random.shuffle(size_type_32x16)
                yield size_type_32x16
                size_type_32x16 = []
            
            # Aggregate batch_size of 16x32
            if (self.data_source[i][7].squeeze()[0].item() == 16 and self.data_source[i][7].squeeze()[1].item() == 32):
                size_type_16x32.append(i)
            # yield them when you achieve batch_size
            if len(size_type_16x32) == self.batch_size:
                random.shuffle(size_type_16x32)
                yield size_type_16x32
                size_type_16x32 = []

            # Aggregate batch_size of 32x8
            if (self.data_source[i][7].squeeze()[0].item() == 32 and self.data_source[i][7].squeeze()[1].item() == 8):                size_type_32x8.append(i)
            # yield them when you achieve batch_size
            if len(size_type_32x8) == self.batch_size:
                random.shuffle(size_type_32x8)
                yield size_type_32x8
                size_type_32x8 = []

            # Aggregate batch_size of 8x32
            if (self.data_source[i][7].squeeze()[0].item() == 8 and self.data_source[i][7].squeeze()[1].item() == 32):
                size_type_8x32.append(i)
            # yield them when you achieve batch_size
            if len(size_type_8x32) == self.batch_size:
                random.shuffle(size_type_8x32)
                yield size_type_8x32
                size_type_8x32 = []

        # if you exit the loop and you didn't achieved batch_size
        # Yield the rest
        if len(size_type_32x8) > 0:
            random.shuffle(size_type_32x8)
            yield size_type_32x8
        if len(size_type_32x16) > 0:
            random.shuffle(size_type_32x16)
            yield size_type_32x16
        if len(size_type_16x16) > 0:
            random.shuffle(size_type_16x16)
            yield size_type_16x16
        if len(size_type_16x32) > 0:
            random.shuffle(size_type_16x32)
            yield size_type_16x32
        if len(size_type_8x32) > 0:
            random.shuffle(size_type_8x32)
            yield size_type_8x32
    def __len__(self):
        return self.batch_size


class SamplerStg6(torch.utils.data.Sampler):
    """! Summary

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, batch_size):
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
    """! Summary

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, batch_size):
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
    """! Summary

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, batch_size):
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

        
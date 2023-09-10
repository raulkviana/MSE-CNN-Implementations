"""@package docstring 

@file demo.py 

@brief Demonstration of the application of the MSE-CNN 

@section libraries_demo Libraries 
- msecnn
- train_model_utils
- cv2
- dataset_utils
- re
- sys
- numpy
- gradio
- torch
- custom_dataset
- PIL
 
@section classes_demo Classes 
- None 

@section functions_demo Functions 
- setup_model()
- int2label(split)
- draw_partition(img, split, cu_pos, cu_size)
- split_fm(cu, cu_pos, split)
- partition_img(img, img_yuv)
- pipeline(img, text)
- main()

@section global_vars_demo Global Variables 
- PATH_TO_COEFFS = "../../../model_coefficients/best_coefficients"
- LOAD_IMAGE_ERROR = "load_image_error.png"
- EXAMPLE_IMGS = ["example_img_1.jpeg", "example_img_2.jpeg"]
- CTU_SIZE = (128, 128)
- FIRST_CU_POS = torch.tensor([0, 0]).reshape(shape=(-1, 2))
- FIRST_CU_SIZE = torch.tensor([64, 64]).reshape(shape=(-1, 2))
- DEV = "cuda" if torch.cuda.is_available() else "cpu"
- QP = 32
- model = None
- COLOR = (0, 247, 255)
- LINE_THICKNESS = 1
- DEFAULT_TEXT_FOR_COORDS = "Insert CTU position in the image..."

@section todo_demo TODO 
- Instead of obtaining the best split, do the thresholding and then split it until you find the right type of split 

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

@section author_demo Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-09-10 21:00:10.225508
"""


# ==============================================================
# Imports
# ==============================================================

import gradio as gr
import cv2 as cv
import sys
import torch
from PIL import Image
import numpy as np
import re

sys.path.append("../")
import msecnn
import dataset_utils as du
import custom_dataset as cd
import train_model_utils as tmu


# ==============================================================
# Constants and Global Variables
# ==============================================================

PATH_TO_COEFFS = "../../../model_coefficients/best_coefficients"
LOAD_IMAGE_ERROR = "load_image_error.png"
EXAMPLE_IMGS = ["example_img_1.jpeg", "example_img_2.jpeg"]
CTU_SIZE = (128, 128)
FIRST_CU_POS = torch.tensor([0, 0]).reshape(shape=(-1, 2))
FIRST_CU_SIZE = torch.tensor([64, 64]).reshape(shape=(-1, 2))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
QP = 32
model = None
COLOR = (0, 247, 255)
LINE_THICKNESS = 1
DEFAULT_TEXT_FOR_COORDS = "Insert CTU position in the image..."

# ==============================================================
# Functions
# ==============================================================

def setup_model():
    """!
    @brief Initializes and load the parameters of the MSE-CNN
    """
    # Initialize model
    stg1_2 = msecnn.MseCnnStg1(device=DEV, QP=QP).to(DEV)
    stg3 = msecnn.MseCnnStgX(device=DEV, QP=QP).to(DEV)
    stg4 = msecnn.MseCnnStgX(device=DEV, QP=QP).to(DEV)
    stg5 = msecnn.MseCnnStgX(device=DEV, QP=QP).to(DEV)
    stg6 = msecnn.MseCnnStgX(device=DEV, QP=QP).to(DEV)
    model = (stg1_2, stg3, stg4, stg5, stg6)

    # Load model coefficients
    model = tmu.load_model_parameters_eval(model, PATH_TO_COEFFS, DEV)

    return model

def int2label(split):
    """!
    @brief Obtain the string that corresponds to an integer value of the split

    @param [in] split: Integer number representing the split tht the model chose
    @param [out] str_split: Name of the corresponding split
    """
    if split == 0:
        return "Non-Split"
    elif split == 1:
        return "Quad-Tree"
    elif split == 2:
        return "Horizontal Binary Tree"
    elif split == 3:
        return "Vertical Binary Tree"
    elif split == 4:
        return "Horizontal Ternary Tree"
    elif split == 5:
        return "Vertical Ternary Tree"
    else:
        return "Something wrong happened!"
    
def draw_partition(img, split, cu_pos, cu_size):
    """!
    @brief Draw partition in image based in the split outputed by the model

    @param [in] img: User's input image
    @param [in] cu_pos: CU position
    @param [in] cu_size: CU size
    @param [in] split: Integer number representing the split that the model chose
    @param [out] str_split: Name of the corresponding split
    """
    # Parameters to draw the lines
    ver_line_length = cu_size[0]
    hor_line_length = cu_size[1]

    if split == 1:
        line1_start = (cu_pos[0], cu_pos[1]+hor_line_length//2)
        line1_end = (cu_pos[0]+ver_line_length, cu_pos[1]+hor_line_length//2)
        line2_start = (cu_pos[0]+ver_line_length//2, cu_pos[1])
        line2_end = (cu_pos[0]+ver_line_length//2, cu_pos[1]+hor_line_length)
        img = cv.line(img, line1_start, line1_end, COLOR, LINE_THICKNESS)
        img = cv.line(img, line2_start, line2_end, COLOR, LINE_THICKNESS)
    elif split == 2:
        line1_start = (cu_pos[0]+ver_line_length//2, cu_pos[1])
        line1_end = (cu_pos[0]+ver_line_length//2, cu_pos[1]+hor_line_length)
        assert line1_start[0]-line1_end[0] == 0 or line1_start[1]-line1_end[1] == 0
        img = cv.line(img, line1_start, line1_end, COLOR, LINE_THICKNESS)
    elif split == 3:
        line1_start = (cu_pos[0], cu_pos[1]+hor_line_length//2)
        line1_end = (cu_pos[0]+ver_line_length, cu_pos[1]+hor_line_length//2)
        # assert line1_start[0]-line1_end[0] == 0 or line1_start[1]-line1_end[1] == 0
        img = cv.line(img, line1_start, line1_end, COLOR, LINE_THICKNESS)
    elif split == 4:
        line1_start = (cu_pos[0]+ver_line_length//3, cu_pos[1])
        line1_end = (cu_pos[0]+ver_line_length//3, cu_pos[1]+hor_line_length)
        line2_start = (cu_pos[0]+(ver_line_length*2)//3, cu_pos[1])
        line2_end = (cu_pos[0]+(ver_line_length*2)//3, cu_pos[1]+hor_line_length)
        img = cv.line(img, line1_start, line1_end, COLOR, LINE_THICKNESS)
        img = cv.line(img, line2_start, line2_end, COLOR, LINE_THICKNESS)
    elif split == 5:
        line1_start = (cu_pos[0], cu_pos[1]+hor_line_length//3)
        line1_end = (cu_pos[0]+ver_line_length, cu_pos[1]+hor_line_length//3)
        line2_start = (cu_pos[0], cu_pos[1]+(hor_line_length*2)//3)
        line2_end = (cu_pos[0]+ver_line_length, cu_pos[1]+(hor_line_length*2)//3)
        img = cv.line(img, line1_start, line1_end, COLOR, LINE_THICKNESS)
        img = cv.line(img, line2_start, line2_end, COLOR, LINE_THICKNESS)
    else:
        raise Exception("Something wrong happened!")
    
    return img


def split_fm(cu, cu_pos, split):
    """!
    @brief Splits feature maps in specific way

    @param [in] cu: Input to the model
    @param [in] cu_pos: Coordinate of the CU
    @param [in] split: Way to split CU
    @param [out] cu_out: New Feature maps
    @param [out] cu_pos_out: Position of the new CUs
    """
    # Initizalize list
    if split == 0:  # Non-split
        cu_out = cu
        cu_pos = [cu_pos]
    
    elif split == 1:  # Quad-tree
        # Split CU and add to list
        cu_1 = torch.split(cu, cu.shape[-2]//2, -2)
        cu_2 = torch.split(cu_1[1], cu_1[1].shape[-1]//2, -1)
        cu_1 = torch.split(cu_1[0], cu_1[0].shape[-1]//2, -1)
        cu_out = cu_1 + cu_2
        cu_pos = [[cu_pos[0], cu_pos[1]], [cu_pos[0], cu_pos[1]+cu.shape[-1]//2],
                  [cu_pos[0]+cu.shape[-2]//2, cu_pos[1]], [cu_pos[0]+cu.shape[-2]//2, cu_pos[1]+cu.shape[-1]//2]]

    elif split == 2:  # HBT
        # Split CU and add to list
        cu_out = torch.split(cu, cu.shape[-2]//2, -2)
        cu_pos = [[cu_pos[0], cu_pos[1]], [cu_pos[0]+cu.shape[-2]//2, cu_pos[1]]]
    
    elif split == 3:  # VBT
        # Split CU and add to list
        cu_out = torch.split(cu, cu.shape[-1]//2, -1)
        cu_pos = [[cu_pos[0], cu_pos[1]], [cu_pos[0], cu_pos[1]+cu.shape[-1]//2]]
                
    elif split == 4:  # HTT
        # Split CU and add to list
        cu_out = torch.split(cu, cu.shape[-2]//3, -2)
        cu_pos = [[cu_pos[0], cu_pos[1]], [cu_pos[0]+cu.shape[-2]//3, cu_pos[1]], [(2*cu.shape[-2])//3+cu_pos[0], cu_pos[1]]]

    
    elif split == 5:  # VTT
        # Split CU and add to list
        cu_out = torch.split(cu, cu.shape[-1]//3, -1)
        cu_pos = [[cu_pos[0], cu_pos[1]], [cu_pos[0], cu_pos[1]+cu.shape[-1]//3], [(2*cu.shape[-1])//3+cu_pos[0], cu_pos[1]]]
    
    else:
        raise Exception("This can't happen! Wrong split mode number: ", str(split))

    if type(cu_out) is tuple:
        if len(cu_out) != 1:
            cu_out = torch.cat(cu_out)
        else:
            cu_out = cu_out[0]

    return cu_out, cu_pos

def partition_img(img, img_yuv):
    """!
    @brief Partitions a full 128x128 CTU and draws the partition in the original image

    TODO: Instead of obtaining the best split, do the thresholding and then split it until you find the right type of split

    @param [in] img: Image in BGR
    @param [in] img_yuv: Image in YUV
    @param [in] stg: Current stage being partitioned
    @param [in] cu_pos: Current stage being partitioned
    @param [in] cu_size: Current stage being partitioned
    @param [out] img: Image in with partitions drawn to it
    """
    global model
    # Stage 1 and 2
    pos_1 = torch.tensor([[0, 0]])
    pos_2 = torch.tensor([[0, 64]])
    pos_3 = torch.tensor([[64, 0]])
    pos_4 = torch.tensor([[64, 64]])
    split_1, CUs_1, ap_1 = model[0](img_yuv, FIRST_CU_SIZE, pos_1)
    split_2, CUs_2, ap_2 = model[0](img_yuv, FIRST_CU_SIZE, pos_2)
    split_3, CUs_3, ap_3 = model[0](img_yuv, FIRST_CU_SIZE, pos_3)
    split_4, CUs_4, ap_4 = model[0](img_yuv, FIRST_CU_SIZE, pos_4)
    all_cus_stg1 = [(split_1, CUs_1, ap_1, (0, 0)), (split_2, CUs_2, ap_2, (0, 64)),
                      (split_3, CUs_3, ap_3, (64, 0)), (split_4, CUs_4, ap_4, (64, 64))]
    img = draw_partition(img, 1, (0, 0), (128, 128))

    # Stage 2: spliting
    for cus_stg1 in all_cus_stg1:
        split_stg1, cu_stg1, ap_stg1, pos_stg1 = cus_stg1
        split_stg1 = tmu.obtain_mode(split_stg1)
        if split_stg1 == 0:
            continue
        # compute new cus
        try:
            cu_out_2, cu_pos_2 = split_fm(cu_stg1, pos_stg1, split_stg1)
        except RuntimeError:
            # Weird partition happened
            continue
        # draw partition to original image
        img = draw_partition(img, split_stg1, pos_stg1, (cu_stg1.shape[-2], cu_stg1.shape[-1]))

        all_cus_stg2 = [(cu_out_2[idx, :, :, :].unsqueeze(0), ap_stg1, cu_pos_2[idx]) for idx in range(cu_out_2.shape[0])]
        
        # Stage 3
        for cus_stg2 in all_cus_stg2:
            cu_stg2, ap_stg2, pos_stg2 = cus_stg2
            pred_stg3, cu_stg3, ap_stg3 = model[1](cu_stg2, ap_stg2)
            pred_stg3 = tmu.obtain_mode(pred_stg3)
            # ap_stg3 = ap_stg3.item()
            if pred_stg3 == 0:
                continue
            # compute new cus
            try:
                cu_out_3, cu_pos_3 = split_fm(cu_stg3, pos_stg2, pred_stg3)
            except RuntimeError:
                # Weird partition happened; skip
                continue
            # draw partition to original image
            img = draw_partition(img, pred_stg3, pos_stg2, (cu_stg3.shape[-2], cu_stg3.shape[-1]))
            
            all_cus_stg3 = [(cu_out_3[idx, :, :, :].unsqueeze(0), ap_stg3, cu_pos_3[idx]) for idx in range(cu_out_3.shape[0])]

            # Stage 4
            for cus_stg3 in all_cus_stg3:
                cu_stg3, ap_stg3, pos_stg3 = cus_stg3
                pred_stg4, cu_stg4, ap_stg4 = model[2](cu_stg3, ap_stg3)
                pred_stg4 = tmu.obtain_mode(pred_stg4)
                # ap_stg4 = ap_stg4.item()
                if pred_stg4 == 0:
                    continue
                
                # compute new cus
                try:
                   cu_out_4, cu_pos_4 = split_fm(cu_stg4, pos_stg3, pred_stg4)
                except RuntimeError:
                    # Weird partition happened; skip
                    continue
                # draw partition to original image
                img = draw_partition(img, pred_stg4, pos_stg3, (cu_stg4.shape[-2], cu_stg4.shape[-1]))
                all_cus_stg4 = [(cu_out_4[idx, :, :, :].unsqueeze(0), ap_stg4, cu_pos_4[idx]) for idx in range(cu_out_4.shape[0])]

                # Stage 5
                for cus_stg4 in all_cus_stg4:
                    cu_stg4, ap_stg4, pos_stg4 = cus_stg4
                    pred_stg5, cu_stg5, ap_stg5 = model[3](cu_stg4, ap_stg4)
                    pred_stg5 = tmu.obtain_mode(pred_stg5)
                    # ap_stg5 = ap_stg5.item()
                    if pred_stg5 == 0:
                        continue
                    # compute new cus
                    try:
                        cu_out_5, cu_pos_5 = split_fm(cu_stg5, pos_stg4, pred_stg5)
                    except RuntimeError:
                        # Weird partition happened; skip
                        continue
                    # draw partition to original image
                    img = draw_partition(img, pred_stg5, pos_stg4, (cu_stg5.shape[-2], cu_stg5.shape[-1]))

                    all_cus_stg5 = [(cu_out_5[idx, :, :, :].unsqueeze(0), ap_stg5, cu_pos_5[idx]) for idx in range(cu_out_5.shape[0])]

                    # Stage 6
                    for cus_stg5 in all_cus_stg5:
                        cu_stg5, ap_stg5, pos_stg5 = cus_stg5
                        pred_stg6, cu_stg6, ap_stg6 = model[4](cu_stg5, ap_stg5)
                        pred_stg6 = tmu.obtain_mode(pred_stg6)
                        # ap_stg6 = ap_stg6.item()
                        if pred_stg6 == 0:
                            continue
                        # draw partition to original image
                        img = draw_partition(img, pred_stg6, pos_stg5, (cu_stg6.shape[-2], cu_stg6.shape[-1]))
    
    return img

def pipeline(img, text):
    """!
    @brief Pipeline to implement the functionalities to demonstrate the potential of the MSE-CNN

    @param [in] img: Image in RGB
    @param [out] mod_img: Modified image with drawings into it in RGB
    @param [out] best_split: Best split (BTV, BTH, TTV, TTH, Non-split, QT)
    """
    global model

    # Obtain coordinates of the CTU
    coords = re.findall(r"\d+", text)
    coords = list(map(lambda x: int(x), coords))

    # In case nothing is submitted, return a default image and text
    if type(img) is type(None) or type(text) is type(None) or text == DEFAULT_TEXT_FOR_COORDS:
        img_error = Image.open(LOAD_IMAGE_ERROR)  # Replace with the path to your first image file
        img_error = np.array(img_error)
        return img_error, "Load the image first and also make sure you specify the position of the CTU!"
    
    # Convert image to appropriate size
    img = img[coords[0]:coords[0]+128, coords[1]:coords[1]+128, :]
    if img.shape[0] % 2 != 0:
        img = img[: img.shape[0]-1, :, :]
    if img.shape[1] % 2 != 0:
        img = img[:, :img.shape[1]-1, :]

    # convert to yuv
    img_yuv = cv.cvtColor(img, cv.COLOR_RGB2YUV_I420)
    # convert to pytorch tensor
    img_yuv = torch.from_numpy(img_yuv)
    # obtain luma channel
    _, ctu_y, _, _ = cd.get_cu_v2(img_yuv, CTU_SIZE, (0, 0), CTU_SIZE)
    # change shape
    ctu_y = torch.reshape(ctu_y, (1, 1, 128, 128)).to(DEV).float()

    # Load model
    model = setup_model()

    # Partition Image
    img = partition_img(img, ctu_y)

    return img, "Partitioned Image"

def main():
    with open("description.md", encoding="utf-8") as f:
        description = f.read()

    in_text_box = gr.Textbox(value=DEFAULT_TEXT_FOR_COORDS, label="Coordinates of CTU", info="You have to provide two numbers indicating the position of the CTU in the image")
    in_image = gr.Image(label="Input image", info="Either use the example image or an image of your choosing")
    out_text_box = gr.Textbox(label="Completion Message")
    out_image = gr.Image(label="Partitioned CTU", info="Result of partitioning using MSE-CNN")

    demo = gr.Interface(fn=pipeline, inputs=[in_image, in_text_box], examples=[[EXAMPLE_IMGS[0], "300, 800"], [EXAMPLE_IMGS[0], "100, 200"], [EXAMPLE_IMGS[1], "600, 925"], [EXAMPLE_IMGS[1], "450, 1600"]], thumbnail="msecnn_model.png",
                        outputs=[out_image, out_text_box], description=description, debug=True, # inbrowser=True,
                        title="MSE-CNN Demo", image="msecnn_model.png")

    demo.launch()


# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    main()

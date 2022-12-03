"""@package docstring 

@file MSECNN.py 

@brief Group of functions and classes that directly contribute for the implementation of the loss function and MSE-CNN
 
@section libraries_MSECNN Libraries 
- numba
- torch
- numpy
- train_model_utils
- math

@section classes_MSECNN Classes 
- CU 
- CU_batch 
- CU_batch_v2 
- QP_half_mask 
- MseCnnStg_1_v2 
- MseCnnStg_x_v2 
- LossFunctionMSE 
- LossFunctionMSE_Ratios 

@section functions_MSECNN Functions 
- init_weights_seq(m)
- init_weights_single(m)
 
@section global_vars_MSECNN Global Variables 
- None 

@section todo_MSECNN TODO 
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

@section author_MSECNN Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2022-12-02 18:21:21.186675
"""

# ==============================================================
# Imports
# ==============================================================

import torch
from torch import nn
import math
import numpy as np
import train_model_utils


# ==============================================================
# Functions
# ==============================================================

def init_weights_seq(m):
    """!
    @brief Initializes a given sequential model with the Xavier Initialization (with uniform distribution)
    @param [in] m: Model to initiliaze the weights
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.PReLU):
        torch.nn.init.xavier_uniform(m.weight)


def init_weights_single(m):
    """!
    @brief Initializes a given layer with the Xavier Initialization (with uniform distribution)
    @param [in] m: Layer to initiliaze the weights
    """

    torch.nn.init.xavier_uniform(m.weight)


# ==============================================================
# Classes
# ==============================================================

class QP_half_mask(nn.Module):
    def __init__(self, QP=32):
        super(QP_half_mask, self).__init__()

        # Initialize varible
        self.QP = QP

    def normalize_QP(self, QP):
        """!
        @brief Normalize the QP value

        @param [in] QP: QP value not normalized
        @param [out] q_tilde: Normalized value of QP
        """

        q_tilde = QP / 51 + 0.5

        return q_tilde

    def forward(self, feature_maps):
        """!
        @brief This function implements the QP half mask operation

        @param [in] feature_maps: Variable with the feature maps
        @param [out] new_feature_maps: Output which differs from the input by half of the feature
        """

        # Normalize QP
        q_tilde = self.normalize_QP(self.QP)

        # Multiply half of the feature_maps by the normalized QP
        new_feature_maps = feature_maps
        new_feature_maps_size = new_feature_maps.size()
        dim_change = 1
        half_num = new_feature_maps_size[dim_change] // 2
        half_1, half_2 = torch.split(new_feature_maps, (half_num, half_num), dim_change)
        half_2 = half_2 * q_tilde
        new_feature_maps = torch.cat((half_1, half_2), dim=dim_change)
        # print("new_feature_maps:",  new_feature_maps[:, 0:(half_num_channels), :, :])

        return new_feature_maps

# Model for stage 1
class MseCnnStg1(nn.Module):

    def __init__(self, device="cpu", QP=32):

        super(MseCnnStg1, self).__init__()
        # Initializing variables
        self.first_simple_conv = nn.Sequential(
            # nn.BatchNorm2D(1), # Consider adding normalization of the input, even knowing it's not mentioned in the paper
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='same', device=device),
            nn.PReLU(num_parameters=1, init=0.2, device=device)
        )  # Simple convolution with activation and padding

        # Conditional convolution stg 1
        self.simple_conv_stg1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device),
            nn.PReLU(num_parameters=1, init=0.2, device=device)
        )  # Simple convolution with activation and padding
        self.simple_conv_no_activation_stg1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device)
        )  # Simple convolution with no activation and padding
        self.simple_conv2_stg1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device),
            nn.PReLU(num_parameters=1, init=0.2, device=device)
        )  # Simple convolution with activation and padding
        self.simple_conv_no_activation2_stg1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device)
        )  # Simple convolution with no activation and padding
        self.activation_PRelu_stg1 = nn.PReLU(num_parameters=1, init=0.2, device=device)  # Parametric
        self.activation_PRelu2_stg1 = nn.PReLU(num_parameters=1, init=0.2, device=device)  # Parametric

        # Conditional convolution stg 2
        self.simple_conv_stg2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device),
            nn.PReLU(num_parameters=1, init=0.2, device=device)
        )  # Simple convolution with activation and padding
        self.simple_conv_no_activation_stg2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device)
        )  # Simple convolution with no activation and padding
        self.simple_conv2_stg2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device),
            nn.PReLU(num_parameters=1, init=0.2, device=device)
        )  # Simple convolution with activation and padding
        self.simple_conv_no_activation2_stg2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device)
        )  # Simple convolution with no activation and padding
        self.activation_PRelu_stg2 = nn.PReLU(num_parameters=1, init=0.2, device=device)  # Parametric
        self.activation_PRelu2_stg2 = nn.PReLU(num_parameters=1, init=0.2, device=device)  # Parametric

        # Initialiaze networks with Xavier initialization
        init_weights_seq(self.first_simple_conv)
        init_weights_seq(self.simple_conv_stg1)
        init_weights_seq(self.simple_conv_no_activation_stg1)
        init_weights_seq(self.simple_conv_stg2)
        init_weights_seq(self.simple_conv_no_activation_stg2)
        init_weights_seq(self.simple_conv2_stg1)
        init_weights_seq(self.simple_conv_no_activation2_stg1)
        init_weights_seq(self.simple_conv2_stg2)
        init_weights_seq(self.simple_conv_no_activation2_stg2)

         # Sub-networks
        self.sub_net = nn.Sequential(QP_half_mask(QP),
                                     nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4, stride=4,
                                               padding='valid', device=device),
                                     nn.PReLU(num_parameters=8, init=0.2, device=device),
                                     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=4, padding='valid', device=device),
                                     nn.PReLU(num_parameters=8, init=0.2, device=device),
                                     QP_half_mask(QP),
                                     nn.Flatten(),
                                     nn.Linear(128, 8, device=device),
                                     nn.PReLU(num_parameters=1, init=0.2, device=device),
                                     nn.Linear(8, 6, device=device),
                                     nn.Softmax(dim=1))


        # Initialize weights
        init_weights_seq(self.sub_net)

    def residual_unit_stg1(self, x, nr):
        """!
        @brief Generic residual unit

        @param [in] x: Input of the network
        @param [in] nr: Number of residual units
        """

        x_shortcut = x  # Copy the initial value

        if nr == 1:
            # Residual unit 1
            x = self.simple_conv_stg1(x)
            x = self.simple_conv_no_activation_stg1(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu_stg1(x)

        elif nr == 2:
            # Residual unit 1
            x = self.simple_conv_stg1(x)
            x = self.simple_conv_no_activation_stg1(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu_stg1(x)

            # Residual unit 2
            x_shortcut = x
            x = self.simple_conv2_stg1(x)
            x = self.simple_conv_no_activation2_stg1(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu2_stg1(x)

        else:
            pass

        return x

    def residual_unit_stg2(self, x, nr):
        """!
        @brief Generic residual unit

        @param [in] x: Input of the network
        @param [in] nr: Number of residual units
        """

        x_shortcut = x  # Copy the initial value

        if nr == 1:
            # Residual unit 1
            x = self.simple_conv_stg2(x)
            x = self.simple_conv_no_activation_stg2(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu_stg2(x)

        elif nr == 2:
            # Residual unit 1
            x = self.simple_conv_stg2(x)
            x = self.simple_conv_no_activation_stg2(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu_stg2(x)

            # Residual unit 2
            x_shortcut = x
            x = self.simple_conv2_stg2(x)
            x = self.simple_conv_no_activation2_stg2(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu2_stg2(x)

        else:
            pass

        return x

    def nr_calc(self, ac, ap):
        """!
        @brief Calculate the number of residual units

        @param [in] ac: Minimum value of the current input axises
        @param [in] ap: Minimum value of the parent input axises
        @param [out] nr: Number of residual units
        """

        nr = 0

        if (ac != 0):

            if (ac == 128):
                nr = 1

            elif (ap != 0):

                if (4 <= ac <= 64):
                    nr = int(math.log2(ap / ac))

                else:
                    raise Exception("ac with invalid number! ac =", ac)

            else:
                raise Exception("ap with invalid number! ap =", ap)

        else:
            raise Exception("ac with invalid number! ac =", ac)

        return nr

    def split(self, cu, coords, sizes, split):
        """!
        @brief Splits feature maps in specific way

        @param [in] cu: Input to the model
        @param [in] coords: Coordinates of the new CUs
        @param [in] coords: Size of the new CUs
        @param [in] split: Way to split CU
        @param [out] cu_out: New Feature maps
        """
        # Initizalize list
        cus_list = []
        for i in range(coords.shape[0]):
            
            if split[i] == 0:  # Non-split
                cus_list.append(cu[i, :, :, :].unsqueeze(dim=0))
            
            elif split[i] == 1:  # Quad-tree
                # Split CU and add to list
                cus_list.append(cu[i, :, coords[i, 0]: coords[i, 0] + sizes[i, 0], coords[i, 1]: coords[i, 1]+sizes[i, 1]].unsqueeze(dim=0))
            
            elif split[i] == 2:  # HBT
                # Split CU and add to list
                cus_list.append(cu[i, :, coords[i, 0]: coords[i, 0] + sizes[i, 0], :].unsqueeze(dim=0))
            
            elif split[i] == 3:  # VBT
                # Split CU and add to list
                cus_list.append(cu[i, :, :, coords[i, 1]: coords[i, 1] + sizes[i, 1]].unsqueeze(dim=0))
                        
            elif split[i] == 4:  # HTT
                # Split CU and add to list
                cus_list.append(cu[i, :, coords[i, 0]: coords[i, 0] + sizes[i, 0], :].unsqueeze(dim=0))
            
            elif split[i] == 5:  # VTT
                # Split CU and add to list
                cus_list.append(cu[i, :, :, coords[i, 1]: coords[i, 1] + sizes[i, 1]].unsqueeze(dim=0))
            
            else:
                raise Exception("This can't happen! Wrong split mode number: ", str(split[i]))

        cu_out = torch.cat(cus_list)

        return cu_out

    def forward(self, cu, sizes=None, coords=None):
        """!
        @brief This functions propagates the input to the output

        @param [in] cu: Input to the model
        @param [out] logits: Vector of raw predictions that a classification model generates
        """

        ## First layer: Overlaping convolution, 3x3 kernel with zero-padding
        cu = self.first_simple_conv(cu)

        ##Conditional Convolution stg 1
        # Number of residual units
        ac = min(cu.shape[-1], cu.shape[-2])  # Getting current minimum axis value
        nr = self.nr_calc(ac, 128)  # Number of residual units, possible values are 0, 1 and 2
        cu = self.residual_unit_stg1(cu, nr)
        # Split CU and get specific
        if coords != None:
            cu = self.split(cu, coords, sizes, np.ones(coords.shape[0]))  # Split in Quad tree

        ##Conditional Convolution stg 2
        # Number of residual units
        ac = min(cu.shape[-1], cu.shape[-2])  # Getting current minimum axis value
        nr = self.nr_calc(ac, 128)  # Number of residual units, possible values are 0, 1 and 2
        cu = self.residual_unit_stg2(cu, nr)

        # Sub-networks
        logits = self.sub_net(cu)

        return logits, cu, ac

# Model for stage 3, 4, 5 and 6
class MseCnnStgX(MseCnnStg1):

    def __init__(self, device="cpu", QP=32):

        super(MseCnnStgX, self).__init__()
        # Hide not needed variables
        # Conditional convolution stg 1
        self.simple_conv_stg1 = None  # Simple convolution with activation and padding
        self.simple_conv_no_activation_stg1 = None  # Simple convolution with no activation and padding
        self.simple_conv2_stg1 = None  # Simple convolution with activation and padding
        self.simple_conv_no_activation2_stg1 = None # Simple convolution with no activation and padding
        self.activation_PRelu_stg1 = None  # Parametric
        self.activation_PRelu2_stg1 = None  # Parametric

        # Conditional convolution stg 2
        self.simple_conv_stg2 = None  # Simple convolution with activation and padding
        self.simple_conv_no_activation_stg2 = None  # Simple convolution with no activation and padding
        self.simple_conv2_stg2 = None  # Simple convolution with activation and padding
        self.simple_conv_no_activation2_stg2 = None  # Simple convolution with no activation and padding
        self.activation_PRelu_stg2 = None  # Parametric
        self.activation_PRelu2_stg2 = None  # Parametric

        self.first_simple_conv = None
        self.sub_net = None

        ## Sub-networks
        # Convolutional layers
        # Min 32
        self.conv_32_64 = nn.Sequential(QP_half_mask(QP),
                                        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(32/8), int(64/8)),
                                        stride=(int(32/8), int(64/8)), padding='valid', device=device))
        self.conv_32_32 = nn.Sequential(QP_half_mask(QP),
                                        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(32/8), int(32/8)),
                                        stride=(int(32/8), int(32/8)), padding='valid', device=device))
        # Min 16
        self.conv_16_16 = nn.Sequential(QP_half_mask(QP),
                                        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(16/4), int(16/4)),
                                        stride=(int(16 / 4), int(16 / 4)), padding='valid', device=device))
        self.conv_16_32 = nn.Sequential(QP_half_mask(QP),
                                        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(16/4), int(32/4)),
                                        stride=(int(16 / 4), int(32 / 4)), padding='valid', device=device))
        self.conv_16_64 = nn.Sequential(QP_half_mask(QP),
                                        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(16/4), int(64/4)),
                                        stride=(int(16/4), int(64/4)), padding='valid', device=device))
        # Min 8
        self.conv_8_8 = nn.Sequential(QP_half_mask(QP),
                                      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(8/2), int(8/2)),
                                      stride=(int(8/2), int(8/2)), padding='valid'))
        self.conv_8_16 = nn.Sequential(QP_half_mask(QP),
                                       nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(8/2), int(16/2)),
                                       stride=(int(8/2), int(16/2)), padding='valid'))
        self.conv_8_32 = nn.Sequential(QP_half_mask(QP),
                                       nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(8/2), int(32/2)),
                                       stride=(int(8/2), int(32/2)), padding='valid'))
        self.conv_8_64 = nn.Sequential(QP_half_mask(QP),
                                       nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(8/2), int(64/2)),
                                       stride=(int(8/2), int(64/2)), padding='valid'))
        # Min  4
        self.conv_4_32 = nn.Sequential(QP_half_mask(QP),
                                       nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(4/2), int(32/2)),
                                       stride=(int(4/2), int(32/2)), padding='valid'))
        self.conv_4_16 = nn.Sequential(QP_half_mask(QP),
                                       nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(4/2), int(16/2)),
                                       stride=(int(4/2), int(16/2)), padding='valid'))
        self.conv_4_8 = nn.Sequential(QP_half_mask(QP),
                                      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(4/2), int(8/2)),
                                      stride=(int(4/2), int(8/2)), padding='valid'))
        self.conv_4_4 = nn.Sequential(QP_half_mask(QP),
                                      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(int(4/2), int(4/2)),
                                      stride=(int(4/2), int(4/2)), padding='valid'))

        # Sub-networks
        self.sub_net_min_32 = nn.Sequential(nn.PReLU(num_parameters=16, init=0.2, device=device),
                                           nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=4,
                                                     padding='valid', device=device),
                                           nn.PReLU(num_parameters=32, init=0.2, device=device),
                                           nn.Conv2d(in_channels=32, out_channels=128, kernel_size=2, stride=2,
                                                     padding='valid', device=device),
                                           nn.PReLU(num_parameters=128, init=0.2, device=device),
                                           QP_half_mask(QP),
                                           nn.Flatten(),
                                           nn.Linear(128, 64, device=device),
                                           nn.PReLU(num_parameters=1, init=0.2, device=device),
                                           nn.Linear(64, 6, device=device),
                                           nn.Softmax(dim=1))


        self.sub_net_min_16 = nn.Sequential(nn.PReLU(num_parameters=16, init=0.2, device=device),
                                           nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2,
                                                     padding='valid', device=device),
                                           nn.PReLU(num_parameters=32, init=0.2, device=device),
                                           nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2,
                                                     padding='valid', device=device),
                                           nn.PReLU(num_parameters=64, init=0.2, device=device),
                                           QP_half_mask(QP),
                                           nn.Flatten(),
                                           nn.Linear(64, 64, device=device),
                                           nn.PReLU(num_parameters=1, init=0.2, device=device),
                                           nn.Linear(64, 6, device=device),
                                           nn.Softmax(dim=1))

        self.sub_net_min_8 = nn.Sequential(nn.PReLU(num_parameters=16, init=0.2, device=device),
                                          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2,
                                                    padding='valid'),
                                          nn.PReLU(num_parameters=32, init=0.2, device=device),
                                          QP_half_mask(QP),
                                          nn.Flatten(),
                                          nn.Linear(32, 16),
                                          nn.PReLU(num_parameters=1, init=0.2, device=device),
                                          nn.Linear(16, 6, device=device),
                                          nn.Softmax(dim=1))

        self.sub_net_min_4 = nn.Sequential(nn.PReLU(num_parameters=16, init=0.2, device=device),
                                          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2,
                                                    padding='valid'),
                                          nn.PReLU(num_parameters=32, init=0.2, device=device),
                                          QP_half_mask(QP),
                                          nn.Flatten(),
                                          nn.Linear(32, 16,  device=device),
                                          nn.PReLU(num_parameters=1, init=0.2, device=device),
                                          nn.Linear(16, 6, device=device),
                                          nn.Softmax(dim=1))


        # Initialize weights
        init_weights_seq(self.sub_net_min_32)
        init_weights_seq(self.conv_32_64)
        init_weights_seq(self.conv_32_32)

        init_weights_seq(self.sub_net_min_16)
        init_weights_seq(self.conv_16_64)
        init_weights_seq(self.conv_16_32)
        init_weights_seq(self.conv_16_16)

        init_weights_seq(self.sub_net_min_8)
        init_weights_seq(self.conv_8_64)
        init_weights_seq(self.conv_8_32)
        init_weights_seq(self.conv_8_16)
        init_weights_seq(self.conv_8_8)

        init_weights_seq(self.sub_net_min_4)
        init_weights_seq(self.conv_4_32)
        init_weights_seq(self.conv_4_16)
        init_weights_seq(self.conv_4_8)
        init_weights_seq(self.conv_4_4)

        # Residual units
        self.simple_conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device),
            nn.PReLU(num_parameters=1, init=0.2, device=device)
        )  # Simple convolution with activation and padding
        self.simple_conv_no_activation = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device)
        )  # Simple convolution with no activation and padding
        self.activation_PRelu = nn.PReLU(num_parameters=1, init=0.2, device=device)  # Parametric Relu activation
                                                                                     # function
        self.simple_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device),
            nn.PReLU(num_parameters=1, init=0.2, device=device)
        )  # Simple convolution with activation and padding
        self.simple_conv_no_activation2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same', device=device)
        )  # Simple convolution with no activation and padding
        self.activation_PRelu2 = nn.PReLU(num_parameters=1, init=0.2, device=device)  # Parametric Relu activation
                                                                                     # function

        # Initialiaze networks with Xavier initialization
        init_weights_seq(self.first_simple_conv)
        init_weights_seq(self.simple_conv)
        init_weights_seq(self.simple_conv_no_activation)
        
    def residual_unit_stg1(self, x, nr):
        """!
        @brief Generic residual unit

        @param [in] x: Input of the network
        @param [in] nr: Number of residual units
        """

        pass

    def residual_unit_stg2(self, x, nr):
        """!
        @brief Generic residual unit

        @param [in] x: Input of the network
        @param [in] nr: Number of residual units
        """

        pass

    def residual_unit(self, x, nr):
        """!
        @brief Generic residual unit

        @param [in] x: Input of the network
        @param [in] nr: Number of residual units
        """

        x_shortcut = x  # Copy the initial value

        if nr == 1:
            # Residual unit 1
            x = self.simple_conv(x)
            x = self.simple_conv_no_activation(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu(x)

        elif nr == 2:
            # Residual unit 1
            x = self.simple_conv(x)
            x = self.simple_conv_no_activation(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu(x)

            # Residual unit 2
            x_shortcut = x
            x = self.simple_conv2(x)
            x = self.simple_conv_no_activation2(x)
            x = torch.add(x_shortcut, x)  # Adding the initial value with the new processed value
            x = self.activation_PRelu2(x)

        else:
            pass

        return x

    def pass_through_subnet(self, x):
        """!
        @brief This functions propagates the it's input through a specific subnetwork depending on the shape of the input

        @param [in] x: Input to the model
        @param [out] logits: Vector of raw predictions that a classification model generates
        """
        try:
            # Obtain input shape: Make sure Height is smaller than Width
            if x.shape[-2] < x.shape[-1]:
                input_shape = (x.shape[-2], x.shape[-1])  # (Height, Width)

            else:
                input_shape = (x.shape[-1], x.shape[-2])  # (Height, Width)

            # Initialize variable
            logits = torch.tensor([])
            if min(input_shape) == 64:

                logits = self.sub_net(x)

            elif min(input_shape) == 32:

                if input_shape == (32, 64):
                    logits = self.conv_32_64(x)
                else:
                    logits = self.conv_32_32(x)

                logits = self.sub_net_min_32(logits)

            elif min(input_shape) == 16:

                if input_shape == (16, 64):
                    logits = self.conv_16_64(x)

                elif input_shape == (16, 32):
                    logits = self.conv_16_32(x)

                else:# input_shape == (16, 16):
                    logits = self.conv_16_16(x)

                logits = self.sub_net_min_16(logits)

            elif min(input_shape) == 8:

                if input_shape == (8, 64):
                    logits = self.conv_8_64(x)

                elif input_shape == (8, 32):
                    logits = self.conv_8_32(x)

                elif input_shape == (8, 16):
                    logits = self.conv_8_16(x)

                else: #input_shape == (8, 8):
                    logits = self.conv_8_8(x)

                logits = self.sub_net_min_8(logits)

            else:# min(input_shape) == 4:
                if input_shape == (4, 32):
                    logits = self.conv_4_32(x)

                elif input_shape == (4, 16):
                    logits = self.conv_4_16(x)

                elif input_shape == (4, 8):
                    logits = self.conv_4_8(x)

                else:
                    logits = self.conv_4_4(x)

                logits = self.sub_net_min_4(logits)

        except:
            raise Exception()

        return logits

    def forward(self, cu, ap, splits, sizes=None, coords=None):
        """!
        @brief This functions propagates the input to the output

        @param [in] cu: Input to the model
        @param [out] logits: Vector of raw predictions that a classification model generates
        """
        
        # Split CU and get specific
        if coords != None:
            cu = self.split(cu, coords, sizes, splits)
            
        ##Conditional Convolution
        # Number of residual units
        ac = min(cu.shape[-1], cu.shape[-2])  # Getting current minimum axis value
        nr = self.nr_calc(ac, ap)  # Number of residual units, possible values are 0, 1 and 2
        cu = self.residual_unit(cu, nr)

        # Transpose if needed
        if not train_model_utils.right_size(cu):
            cu_transposed = torch.clone(torch.transpose(cu, -1, -2))
        else:
            cu_transposed = torch.clone(cu)

        # Sub-networks
        logits = self.pass_through_subnet(cu_transposed)

        return logits, cu, ac

class LossFunctionMSE(nn.Module):
    def __init__(self, use_mod_cross_entropy=True, beta=1):
        super(LossFunctionMSE, self).__init__()
        self.beta = beta
        self.use_mod_cross_entropy = use_mod_cross_entropy
        # Constants
        self.MAX_RD = 1E10 # Valor encontrado no dataset: 1.7E304
        self.MAX_LOSS = 20 # float("inf")
        self.last_loss = -1
        self.last_pred = -1
        self.last_RD = -1

    def get_proportion_CUs(self, labels):
        """!
        @brief This function returns the proportion of CU's for all the types of split mode

        @param [in] labels: Ground truth tensor
        @param [out] p_m: Tensor with the proportions
        """

        p_m = torch.sum(labels, dim=0)
        p_m = torch.reshape(p_m / torch.sum(p_m), shape=(1, -1))

        return p_m

    def get_min_RDs(self, RDs):
        """!
        @brief Obtain the lowest value that isnt zero from RDs tensor

        @param [in] RDs: Tensor with RDs
        @param [out] min_RD: Lowest value of RDs that isnt zero
        """

        clone_RDs = RDs.clone()
        mask = (clone_RDs == 0)  # Mask with the values equal to zero
        clone_RDs[mask] = float('inf')  # Idx where there are zeros are substituted by infinity
        min_RD = torch.reshape(torch.min(clone_RDs, dim=1)[0], shape=(-1, 1))  # Get minimum values of RD

        return min_RD

    def remove_values_lower(self, tensor, max_val, subst_val):
        """!
        @brief Remove values from tensor that are lower than a given value

        @param [in] tensor: Tensor with values
        @param [in] max_val: Threshold val
        @param [in] subst_val: Max value to replace the others
        @param [out] tensor: New tensor
        """

        mask = (tensor < max_val)  # Mask with the values equal to zero
        tensor[mask] = subst_val  # Idx where there are zeros are substituted by infinity

        return tensor

    def remove_inf_values(self, tensor):
        """!
        @brief Remove values from tensor that are inf and make them zeros

        @param [in] tensor: Tensor with values
        @param [out] tensor: New tensor
        """

        clone_RDs = tensor.clone()
        mask = (clone_RDs == float('inf'))  # Mask with the values equal to zero
        clone_RDs[mask] = 0  # Idx where there are zeros are substituted by infinity

        return clone_RDs

    def remove_zero(self, RDs):
        """!
        @brief Substitutes the zeros values for big RD values

        @param [in] RDs: Tensor with RDs
        @param [out] RDs: Tensor With Max values added
        """

        clone_RDs = RDs.clone()
        mask = (clone_RDs == 0.0) # Mask with the values equal to zero
        clone_RDs[mask] = self.MAX_RD

        return clone_RDs
    
    def remove_values_above(self, RDs, max_val):
        """!
        @brief Substitutes values above MAX_RD for the MAX_RD

        @param [in] RDs: Tensor with RDs
        @param [in] max_val: Max value to add
        @param [out] clone_RDs: Tensor With Max values added
        """

        clone_RDs = RDs.clone()
        mask = (clone_RDs >= max_val) # Mask with the values equal to zero
        clone_RDs[mask] = max_val

        return clone_RDs

    def forward(self, pred, labels, RD):
        """!
        @brief This function implements the loss function

        @param [in] pred: Predictions made by the model
        @param [in] labels: Ground-truth tensor
        @param [in] RD: Rate distortion tensor
        @param [out] loss: Vector of raw predictions that a classification model generates
        """

        # Cross Entropy loss
        if self.use_mod_cross_entropy:
            # Loss
            loss_CE = torch.mul(torch.log(torch.add(pred, 0.000000000000000001)), labels)
            loss_CE = torch.sum(loss_CE, dim=1)
            loss_CE = -torch.mean(loss_CE, dim=0)

        else:
            # Cross Entropy loss
            labels_mod = torch.squeeze(train_model_utils.obtain_mode(labels))
            loss_CE = nn.CrossEntropyLoss()(pred, labels_mod)  # 6 classes

        #print("Cross entropy loss:", loss_CE)

        # Rate distortion
        min_RDs = self.get_min_RDs(RD)  # Get minimum values of RD
        #min_RDs = torch.log10(min_RDs)

        # print("min_RDs", min_RDs)

        # Substitute inf RD values
        RD_mod = self.remove_inf_values(RD)

        # Replace zero values for a big number
        RD_mod = self.remove_zero(RD_mod)
        #RD_mod = torch.log10(RD_mod)

        # Replace high values for a smaller number
        RD_mod = self.remove_values_above(RD_mod, self.MAX_RD)

        # # Remove last 4 classes
        # RD_mod = RD_mod[:,:-4]
        # pred = pred[:,:-4]

        # Compute loss
        # Paper loss function
        loss_RD = torch.mul(pred, torch.sub(torch.div(RD_mod, min_RDs), 1))
        # Modified
        #loss_RD = -torch.mul(torch.log(pred), torch.sub(torch.div(RD_mod, min_RDs), 1))

        # Proportions way of calculating loss function
        #RD_proportions = torch.div(RD_mod, torch.reshape(torch.sum(RD_mod, dim=1), shape=(-1,1)))  # Compute RD proportions 
        #RD_proportions_mod = torch.pow(torch.div(1, torch.sub(1, RD_proportions)), 4)  # Apply subtraction and use power of a constant for scaling 
        #loss_RD = torch.mul(pred, RD_proportions_mod)
        #loss_RD = -torch.mul(torch.log(pred), RD_proportions_mod)

        # Normalazing with max RD value and calculate loss function
        #RD_maxs = torch.reshape(torch.max(RD_mod, dim=1)[0], shape=(-1, 1))
        #RD_mod2 = torch.div(RD_mod, RD_maxs)
        #loss_RD = torch.mul(pred, RD_mod2)
        #loss_RD = -torch.mul(torch.log(pred), RD_mod2)

        # Loss functio that uses means
        #means_losses = torch.mean(RD_mod, dim=1, keepdim=True)
        #RD_mod1 = RD_mod.clone()
        #RD_mod1 [RD_mod > means_losses] = 5
        #RD_mod1 [RD_mod <= means_losses] = 0
        #loss_RD = torch.mul(pred, RD_mod1)

        # Remove negative
        loss_RD = self.remove_values_lower(loss_RD, 0, 0)

        # Mean loss
        loss_RD = torch.sum(loss_RD, dim=1)
        loss_RD = torch.mean(loss_RD, dim=0)
        if loss_RD.item() > self.MAX_LOSS:
            # Reduce loss
            temp = torch.div(self.MAX_LOSS, loss_RD)
            loss_RD = torch.mul(loss_RD, temp)

        #print("Rate Distortion loss:", loss_RD)

        # Total loss
        loss = torch.add(loss_CE, torch.mul(loss_RD, self.beta))
        #loss = loss_RD

        # Exceptions
        if torch.isnan(loss):
            raise Exception("Loss can not be equal to nan!! Last loss was: ", str(self.last_loss))

        elif torch.isinf(loss):
            raise Exception("Loss is infinite!! Last loss was: ", str(self.last_loss))

        self.last_loss = loss
        #self.last_RD = [RD, RD_mod, RD_proportions, RD_proportions_mod]
        self.last_pred = pred

        return loss, loss_CE, loss_RD

class LossFunctionMSE_Ratios(nn.Module):
    # This version uses penalty weights for less represented classes
    def __init__(self, pm, use_mod_cross_entropy=True, beta=1, alpha=0.5):
        super(LossFunctionMSE_Ratios, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.pm = pm
        self.use_mod_cross_entropy = use_mod_cross_entropy
        # Constants
        self.MAX_RD = 1E10 # Valor encontrado no dataset: 1.7E304
        self.MAX_LOSS = 20 # float("inf")
        self.last_loss = -1
        self.last_pred = -1
        self.last_RD = -1

    def get_proportion_CUs(self, labels):
        """!
        @brief This function returns the proportion of CU's for all the types of split mode

        @param [in] labels: Ground truth tensor
        @param [out] p_m: Tensor with the proportions
        """

        p_m = torch.sum(labels, dim=0)
        p_m = torch.reshape(p_m / torch.sum(p_m), shape=(1, -1))

        return p_m

    def get_min_RDs(self, RDs):
        """!
        @brief Obtain the lowest value that isnt zero from RDs tensor

        @param [in] RDs: Tensor with RDs
        @param [out] min_RD: Lowest value of RDs that isnt zero
        """

        clone_RDs = RDs.clone()
        mask = (clone_RDs == 0)  # Mask with the values equal to zero
        clone_RDs[mask] = float('inf')  # Idx where there are zeros are substituted by infinity
        min_RD = torch.reshape(torch.min(clone_RDs, dim=1)[0], shape=(-1, 1))  # Get minimum values of RD

        return min_RD

    def remove_values_lower(self, tensor, max_val, subst_val):
        """!
        @brief Remove values from tensor that are lower than a given value

        @param [in] tensor: Tensor with values
        @param [in] max_val: Threshold val
        @param [in] subst_val: Max value to replace the others
        @param [out] tensor: New tensor
        """

        mask = (tensor < max_val)  # Mask with the values equal to zero
        tensor[mask] = subst_val  # Idx where there are zeros are substituted by infinity

        return tensor

    def remove_inf_values(self, tensor):
        """!
        @brief Remove values from tensor that are inf and make them zeros

        @param [in] tensor: Tensor with values
        @param [out] tensor: New tensor
        """

        clone_RDs = tensor.clone()
        mask = (clone_RDs == float('inf'))  # Mask with the values equal to zero
        clone_RDs[mask] = 0  # Idx where there are zeros are substituted by infinity

        return clone_RDs

    def remove_zero(self, RDs):
        """!
        @brief Substitutes the zeros values for big RD values

        @param [in] RDs: Tensor with RDs
        @param [out] RDs: Tensor With Max values added
        """

        clone_RDs = RDs.clone()
        mask = (clone_RDs == 0.0) # Mask with the values equal to zero
        clone_RDs[mask] = self.MAX_RD

        return clone_RDs
    
    def remove_values_above(self, RDs, max_val):
        """!
        @brief Substitutes values above MAX_RD for the MAX_RD

        @param [in] RDs: Tensor with RDs
        @param [in] max_val: Max value to add
        @param [out] clone_RDs: Tensor With Max values added
        """

        clone_RDs = RDs.clone()
        mask = (clone_RDs >= max_val) # Mask with the values equal to zero
        clone_RDs[mask] = max_val

        return clone_RDs

    def forward(self, pred, labels, RD):
        """!
        @brief This function implements the loss function

        @param [in] pred: Predictions made by the model
        @param [in] labels: Ground-truth tensor
        @param [in] RD: Rate distortion tensor
        @param [out] loss: Vector of raw predictions that a classification model generates
        """

        # Cross Entropy loss
        if self.use_mod_cross_entropy:
            # Loss
            loss_CE = torch.mul(torch.log(torch.add(pred, 0.000000000000000001)), labels)
            loss_CE = torch.mul(torch.pow(1/self.pm, self.alpha), loss_CE)
            loss_CE = torch.sum(loss_CE, dim=1)
            loss_CE = -torch.mean(loss_CE, dim=0)

        else:
            # Cross Entropy loss
            labels_mod = torch.squeeze(train_model_utils.obtain_mode(labels))
            loss_CE = nn.CrossEntropyLoss()(pred, labels_mod)  # 6 classes

        #print("Cross entropy loss:", loss_CE)

        # Rate distortion
        min_RDs = self.get_min_RDs(RD)  # Get minimum values of RD
        #min_RDs = torch.log10(min_RDs)

        # print("min_RDs", min_RDs)

        # Substitute inf RD values
        RD_mod = self.remove_inf_values(RD)

        # Replace zero values for a big number
        RD_mod = self.remove_zero(RD_mod)
        #RD_mod = torch.log10(RD_mod)

        # Replace high values for a smaller number
        RD_mod = self.remove_values_above(RD_mod, self.MAX_RD)

        # # Remove last 4 classes
        # RD_mod = RD_mod[:,:-4]
        # pred = pred[:,:-4]

        # Compute loss
        # Paper loss function
        loss_RD = torch.mul(pred, torch.sub(torch.div(RD_mod, min_RDs), 1))
        # Modified
        #loss_RD = -torch.mul(torch.log(pred), torch.sub(torch.div(RD_mod, min_RDs), 1))

        # Proportions way of calculating loss function
        #RD_proportions = torch.div(RD_mod, torch.reshape(torch.sum(RD_mod, dim=1), shape=(-1,1)))  # Compute RD proportions 
        #RD_proportions_mod = torch.pow(torch.div(1, torch.sub(1, RD_proportions)), 4)  # Apply subtraction and use power of a constant for scaling 
        #loss_RD = torch.mul(pred, RD_proportions_mod)
        #loss_RD = -torch.mul(torch.log(pred), RD_proportions_mod)

        # Normalazing with max RD value and calculate loss function
        #RD_maxs = torch.reshape(torch.max(RD_mod, dim=1)[0], shape=(-1, 1))
        #RD_mod2 = torch.div(RD_mod, RD_maxs)
        #loss_RD = torch.mul(pred, RD_mod2)
        #loss_RD = -torch.mul(torch.log(pred), RD_mod2)

        # Loss functio that uses means
        #means_losses = torch.mean(RD_mod, dim=1, keepdim=True)
        #RD_mod1 = RD_mod.clone()
        #RD_mod1 [RD_mod > means_losses] = 5
        #RD_mod1 [RD_mod <= means_losses] = 0
        #loss_RD = torch.mul(pred, RD_mod1)

        # Remove negative
        loss_RD = self.remove_values_lower(loss_RD, 0, 0)

        # Mean loss
        loss_RD = torch.sum(loss_RD, dim=1)
        loss_RD = torch.mean(loss_RD, dim=0)
        if loss_RD.item() > self.MAX_LOSS:
            # Reduce loss
            temp = torch.div(self.MAX_LOSS, loss_RD)
            loss_RD = torch.mul(loss_RD, temp)

        #print("Rate Distortion loss:", loss_RD)

        # Total loss
        loss = torch.add(loss_CE, torch.mul(loss_RD, self.beta))
        #loss = loss_RD

        # Exceptions
        if torch.isnan(loss):
            raise Exception("Loss can not be equal to nan!! Last loss was: ", str(self.last_loss))

        elif torch.isinf(loss):
            raise Exception("Loss is infinite!! Last loss was: ", str(self.last_loss))

        self.last_loss = loss
        #self.last_RD = [RD, RD_mod, RD_proportions, RD_proportions_mod]
        self.last_pred = pred

        return loss, loss_CE, loss_RD

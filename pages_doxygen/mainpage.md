

# MSE-CNN Implementation

<div align="center">
  <img src="../../img/msecnn_model.png" width=800 />
</div>

Code database with an implementation of MSE-CNN [1]. Besides the code, the dataset and coefficients obtained after training are provided.


```python
>>> import torch
>>> import msecnn
>>> import train_model_utils
>>>
>>> # Initialize parameters
>>> path_to_folder_with_model_params = "model_coefficients/best_coefficients"
>>> device = "cuda:0"
>>> qp = 32  # Quantisation Parameter
>>> 
>>> # Initialize Model
>>> stg1_2 = msecnn.MseCnnStg1(device=device, QP=qp).to(device)
>>> stg3 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
>>> stg4 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
>>> stg5 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
>>> stg6 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
>>> model = (stg1_2, stg3, stg4, stg5, stg6)
>>>
>>> model = train_model_utils.load_model_parameters_eval(model, path_to_folder_with_model_params, device)
>>>
>>> # Loss function
>>> loss_fn = msecnn.LossFunctionMSE()
>>> 
>>> # Path to labels
>>> l_path_val = "example_data/stg2"
>>>
>>> # Random CTU and labels
>>> CTU = torch.rand(1, 1, 128, 128).to(device)
>>> CTU
tensor([[[[0.9320, 0.6777, 0.4490,  ..., 0.0413, 0.6278, 0.5375],
          [0.3544, 0.5620, 0.8339,  ..., 0.6420, 0.2527, 0.3104],
          [0.0555, 0.4991, 0.9972,  ..., 0.3898, 0.1169, 0.1661],
          ...,
          [0.9452, 0.3566, 0.9825,  ..., 0.3941, 0.7534, 0.8656],
          [0.3839, 0.8459, 0.4369,  ..., 0.9569, 0.2609, 0.6421],
          [0.1734, 0.7182, 0.8074,  ..., 0.2122, 0.7573, 0.2492]]]])
>>> cu_pos = torch.tensor([[0, 0]]).to(device)
>>> cu_size = torch.tensor([[64, 64]]).to(device)  # Size of the CU of the second stage
>>> split_label = torch.tensor([[1]]).to(device)
>>> RDs = torch.rand(1, 6).to(device) * 10_000
>>> RDs
tensor([[1975.6646, 2206.7600, 1570.3577, 3570.9478, 6728.2612,  527.9994]])
>>> # Compute prediction for stages 1 and 2
>>> # Stage 1 and 2
>>> pred1_2, CUs, ap = model[0](CTU, cu_size, cu_pos)  # Pass CU through network
>>> pred1_2
tensor([[9.9982e-01, 1.8124e-04, 9.9010e-21, 5.9963e-29, 1.9118e-24, 1.0236e-25]],
       grad_fn=<SoftmaxBackward0>)
>>> CUs.shape
torch.Size([1, 16, 64, 64])
>>> 
>>> # Compute the loss
>>> loss, loss_CE, loss_RD = loss_fn(pred1_2, split_label, RDs)
>> loss
tensor(177.1340, grad_fn=<AddBackward0>)
>> loss_CE
tensor(174.3921, grad_fn=<NegBackward0>)
>> loss_RD
tensor(2.7419, grad_fn=<MeanBackward1>)
```
<br>

- [MSE-CNN Implementation](#mse-cnn-implementation)
  - [1. Introduction](#1-introduction)
  - [2. Theorectical Background](#2-theorectical-background)
    - [2.1 Partitioning in VVC](#21-partitioning-in-vvc)
    - [2.2 MSE-CNN](#22-mse-cnn)
      - [2.2.1 Architecture](#221-architecture)
      - [2.2.2 Loss Function](#222-loss-function)
      - [2.2.3 Training](#223-training)
      - [2.2.4 Implementation remarks](#224-implementation-remarks)
  - [3. Dataset](#3-dataset)
  - [4. Results](#4-results)
    - [4.1 F1-score, Recall and Precision with test data](#41-f1-score-recall-and-precision-with-test-data)
    - [4.2 Confusion matrices](#42-confusion-matrices)
      - [4.2.1 Stages 2 and 3](#421-stages-2-and-3)
      - [4.2.2 Stages 4 and 5](#422-stages-4-and-5)
      - [4.2.3 Stage 6](#423-stage-6)
    - [4.3 Y-PSNR, Complexity Reduction and Bitrate with test data](#43-y-psnr-complexity-reduction-and-bitrate-with-test-data)
  - [5. Relevant Folders and files](#5-relevant-folders-and-files)
    - [5.1 Folders](#51-folders)
    - [5.2 Files in src folder](#52-files-in-src-folder)
  - [6. Installation of dependencies](#6-installation-of-dependencies)
    - [Requirements](#requirements)
    - [Package Distributions](#package-distributions)
  - [7. Demo](#7-demo)
  - [8. Contributions](#8-contributions)
  - [9. License](#9-license)
  - [10. TODO](#10-todo)
  - [11. References](#11-references)


## 1. Introduction

The emergence of new technologies that provide creative audiovisual experiences, such as 360-degree films, virtual reality, augmented reality, 4K, 8K UHD, 16K, and also with the rise of video traffic on the web, shows the current demand for video data in the modern world. Because of this tension, Versatile Video Coding (VVC) was developed due to the the necessity for the introduction of new coding standards. Despite the advancements achieved with the introduction of this standard, its complexity has increased very much. The new partitioning technique is responsible for majority of the increase in encoding time. This extended duration is linked with the optimization of the Rate-Distortion cost (RD cost). Although VVC offers higher compression rates, the complexity of its encoding is high.

<div align="center">
  <img src="../../funny_memes_about_this_work/72rrr9.jpg" width=300 />
  <p>Fig. 1: VVC Complexity</p>
</div>

In light of this, the Multi-Stage Exit Convolutional 
Neural Nework (MSE-CNN) was developed. This Deep Learning-based model is organised in a sequential structure with several stages. Each stage, which represents a different partition depth, encompasses a set of layers for extracting features from a Coding Tree Unit (CTU) and deciding how to partition it. Instead of using recursive approaches to determine the optimal way to fragment an image, this model allows VVC to estimate the most appropriate way of doing it. **This work presents a model of the MSE-CNN that employs training procedures distinct from the original implementation of this network, as well as the ground-thruth to train and validate the model and an interpretation of the work done by the MSE-CNN’s original creators**.

<div align="center">
  <img src="../../img/funny_memes_about_this_work/72roie.jpg" width=400 />
  <p>Fig. 2: MSE-CNN benefits</p>
</div>

## 2. Theorectical Background

### 2.1 Partitioning in VVC

The key objective of partitioning is to divide frames into pieces in a way that results in a
reduction of the RD cost. To achieve a perfect balance of quality and bitrate, numerous image
fragments combinations must be tested, which is computationally expensive. Due to the intensive
nature of this process, a high compression rate can be attained. Partitioning contributes heavily
to both the complexity and compression gains in VVC. H.266 (VVC), organize a video sequence in many frames that are divided into smaller pieces. First, pictures are split into coding tree units (CTUs), and then they are divided into coding units (CUs). For the luma channel, the largest CTU size in
VVC is 128x128 and the smallest is 4x4. In VVC, a quad-tree (QT) is initially applied to the CTUs in the first level, and then a quad-tree with nested multi-type tree (QMTT) is applied recursively. 

<div align="center">
  <img src="../../img/vvc_parti_real.png" />
  <p>Fig. 3: Types of partitions in VVC</p>
</div>

This innovation makes it possible to split CUs in different rectangle forms. Splitting a CU into:

* three rectangles with a ratio of 1:2:1 results in a ternary tree (TT), with the center rectangle being half the size of the original CU; when applied horizontally it is called a horizontal ternary tree (HTT), and vertical ternary tree (VTT) when it is done vertically. 
* two rectangles results in a binary tree (BT)partition, a block with two symmetrical structures; like in the case of the TT, depending on the way the split is done, it can be called either
a vertical binary tree (VBT) or a horizontal binary tree (HBT).

The association of BT and TT is named a multi-type tree (MTT). The introduction of BT and TT partitions enables the creation of various new types of forms, with heights and widths that can be a combination between 128, 64, 32, 16, 8 and 4. The increased number of possible CUs boosts the ability of the codec to fragment
an image more efficiently, allowing better predictions and higher compressing abilities. Although this standard now have these advantages, as a downside it takes longer to encode.

<div align="center">
  <img src="../../img/partitioning_image.png" />
  <p>Fig. 4: Partitioning in VVC</p>
</div>


### 2.2 MSE-CNN

Multi-Stage Exit Convolutional Neural Network (MSE-CNN) is a DL model that seeks to forecast CUs in a waterfall architecture (top-down manner), it integrates . This structure takes a CTU as input, extracts features from it, splits the CU into one of at most six possible partitions (Non-split, QT, HBT, VBT, HTT, and VTT), and then sends it to the next stage. This model has CTUs as inputs in the first stage, either in the chroma or luma channel, and feature maps in the subsequent stages. Furthermore, it generates feature maps and a split decision at each level. In the event that one of the models returns the split decision as Non-Split, the partitioning of the CU is ended immediately.

**Note**: Details about how to load model coefficients can be found [here](modelcoefpage.md).

#### 2.2.1 Architecture

This model is composed by the following blocks:

* Initially, this model adds more channels to the input of this network to create more features
from it; this is accomplished by utilising simple convolutional layers. 

<div align="center">
  <img src="../../img/over_con_block.drawio.png" />
  <p>Fig. 5: Overlapping convolution layer</p>
</div>

* To extract more characteristics from the data, the information is then passed through a series of convolutional layers; these layers were named Conditional Convolution. 
  
<div align="center">
  <img src="../../img/resnet_mse.png" />
  <p>Fig. 6: Conditional Convolution</p>
</div>


* At the end, a final layer is employed to determine the
optimal manner of partitioning the CU. This layer is a blend of fully connected and convolutional
layers.

<div align="center">
  <img src="../../img/sub_networks.png" />
  <p>Fig. 7: Sub-networks</p>
</div>

<span style="text-decoration: underline">Note</span>: For more details regarding these layers check [1]
 

#### 2.2.2 Loss Function

The loss developed for the MSE-CNN is the result of two other functions, as defined in the
following expression:

$$ L = L_{CE}+\beta L_{RD}$$

In the above equation, $\beta$ is a real number to adjust the influence of the $L_{RD}$ loss. The first member of this loss function is a modified Cross-Entrotopy loss, developed to solve imbalanced dataset issues:

$$L_{CEmod} = -\frac{1}{N}\sum_{n=1}^N \sum_{m\varepsilon Partitions}(\frac{1}{p_m})^\alpha y_{n, m}\log(\hat{y}_{n, m})$$

<div align="center">
  <sub> Eq. 1: In this equation "n" is the batch number, "m" is the corresponding partition (0 (Non-Split), 1 (QT), 2 (HBT), 3 (VBT), 4 (VTT), 5 (HTT)), "N" is the total number of batches and alpha is a parameter to configure the penalties for the less represented classes </sub> 
</div>

<br>
<br>

Concerning the second member of the MSE-CNN loss function, this constituent gives the network the ability to also make predictions based on the RD Cost.

$$L_{RD} = \frac{1}{N}\sum_{n=1}^N \sum_{m\varepsilon Partitions}\hat{y}_{n, m}\frac{r_{n, m}}{r_{n, min}}-1$$

In the above equation, the RD costs $r_{n, m}$ uses the same notation for "n" and "m" as the previous equation. Regarding $r_{n ,min}$, it is the minimal RD cost for the nth CU among all split modes and 
$$\frac{r_{n, m}}{r_{n, min}} - 1$$
is a normalised RD cost. As a relevant note, $r_{n, min}$ is equal to the RD cost of the best partition mode. Consequently, the result of

<div align="center">
  <img src="../../img/formula.png" />
</div>

ensures that CU's partitions with greater erroneously predicted probability values or greater RD cost values $r_{n, m}$ are more penalised. In $\frac{r_{n, m}}{r_{n, min}} - 1$, the ideal partition has a normalised RD cost of zero, but the other partitions do not. Therefore, the only way for the loss to equal zero is if the probability for all other modes also equals zero. Consequently, the learning algorithm must assign a greater probability to the optimal split mode while reducing the probabilities for the rest. **Experimentally it was verified that this function wasn't able to contribute to the training of the MSE-CNN, this contradicted the remarks made in [1]**.

#### 2.2.3 Training

The strategy used to train the MSE-CNN was very similar to the one used in [1]. The first parts of the model to be trained were the first and second stages, in which 64x64 CUs were passed through the second depth. Afterwards, transfer learning was used to pass certain coefficients of the second stage to the third. Then, the third stage was trained with 32x32 CUs flowing through it. After this step, a similar process was done to the following stages. It is worth noting that, beginning with stage 4, various CUs forms are at the models' input. This means that these stages were fed different kinds of CUs.  

<div align="center">
  <img src="../../img/training_steps.png" width=300/>
  <p>Fig. 8: Training flow used</p>
</div>

At the end of training, 6 models were obtained one for each partitioning depth in the luma channel. Although models for the luma and chroma channels could be created for all the shapes of CUs that are possible, rather than just for each depth, only six were trained for the sake of assessing the model behaviour in a simpler and more understandable configuration.

#### 2.2.4 Implementation remarks

Due to the deterministic nature of the first stage, where CTUs are always partitioned with a QT, it was implemented together with the second stage. If it was done separately, the training for the first two stages would have to be done at the same time. Consequently, two distinct optimisers would need to be employed, which could result in unpredictable training behaviour. <br>

<div align="center">
  <img src="../../img/subnet_min_32_1.drawio.png" width=300/>
  <img src="../../img/subnet_min_32_2.drawio.png" width=300/>
  <p>Fig. 9: 32 minimum axis size sub-networks</p>
</div>

When implementing the sub-networks on code, those that were meant to cater for varying CU sizes were further implemented separately. For example, in the case of the sub-network utilised when the minimum width or height is 32, two variants of the first two layers were built. This was done because 64x32 and 32x32 CUs can flow across this block. Because of this, the first two layers were implemented separately from the entire block. Then, they were used in conjunction with the remaining layers based on the dimensions of the input CU. The same procedures were followed for the other types of sub-networks.

When the network was being trained, some of the RD costs from the input data had very high values. Consequently, the RD loss function value skyrocketed, resulting in extremely huge gradients during training. As a result, the maximum RD cost was hard coded at $10^{10}$. This amount is large enough to be more than the best partition's RD cost and small enough to address this issue. 

## 3. Dataset

Please see this [page](dataset.md) to understand better the dataset and also access it. To see example data go to follow [this](exampledatapage.md).

## 4. Results

Since it was verified that the Rate-Distortion Loss. $L_{RD}$, could contribute for better results, the metrics presented here were obtained with a model trained only with the modified cross-entropy loss.

### 4.1 F1-score, Recall and Precision with test data

| Stage | F1-Score | Recall | Precision |
|-------|----------|--------|-----------|
| Stage 2 | 0.9111 | 0.9111 | 0.9112 |
| Stage 3 | 0.5624 | 0.5767 | 0.5770 |
| Stage 4 | 0.4406 | 0.4581 | 0.4432 |
| Stage 5 | 0.5143 | 0.5231 | 0.5184 |
| Stage 6 | 0.7282 | 0.7411 | 0.7311 |

Results with weighted average for F1-score, recall and precision.

### 4.2 Confusion matrices

#### 4.2.1 Stages 2 and 3
<div align="center">
  <img src="../../img/conf_mat_val_stg2.png" width=300 />
  <img src="../../img/conf_mat_val_v2_stg3.png" width=300 />
  <p>Fig. 10: Confusion matrix results with the testing data in stages 2 and 3</p>
</div>

#### 4.2.2 Stages 4 and 5
<div align="center">
  <img src="../../img/conf_mat_val_stg4.png" width=300 />
  <img src="../../img/conf_mat_val_stg5.png" width=300 />
  <p>Fig. 11: Confusion matrix results with the testing data in stages 4 and 5</p>
</div>

#### 4.2.3 Stage 6
<div align="center">
  <img src="../../img/conf_mat_val_stg6.png" width=300 />
  <p>Fig. 12: Confusion matrix results with the testing data in stage 6</p>
</div>



### 4.3 Y-PSNR, Complexity Reduction and Bitrate with test data

| Metric | VTM-7.0 | VTM-7.0+Model | Gain |
|-------|----------|--------|-----------|
| Bitrate | 3810.192 kbps | 4069.392 kbps | 6.80% |
| Y-PSNR | 35.7927 dB  | 35.5591 dB | -0.65% |
| Complexity | 1792.88 s  | 1048.95 s | -41.49% |

**These results were obtained with the "medium" configuration for the multi-thresholding method.**

## 5. Relevant Folders and files

### 5.1 Folders

| Folder | Description |
|--------|-------------|
| [dataset](/dataset) | This folder contains all of the dataset and all of the data that was processed in order to obtain it |
| [example_data](/example_data) | Here you can find some example data that it is used for the scripts in usefull_scripts folder|
| [model_coefficients](/model_coefficients) | The last coefficient obtained during training, as well as the best one in terms of the best F1-score obtained in testing data |
| [src](/src) | Source code with the implementation of the MSE-CNN and also useful code and examples |


### 5.2 Files in src folder

| Files | Description |
|--------|-------------|
| constants.py | Constant values used in other python files |
| custom_dataset.py | Dataset class to handle the files with the ground-thruth information, as well as other usefull classes to work together with the aforementioned class |
| dataset_utils.py | Functions to manipulate and process the data, also contains functions to interact with YUV files |
| msecnn.py | MSE-CNN and Loss Function classes implementation |
| train_model_utils.py | Usefull functions to be used during training or evaluation of the artificial neural network |
| utils.py | Other functions that are usefull not directly to the model but for the code implementation itself |

## 6. Installation of dependencies
In order to explore this project, it is needed to first install of the libraries used in it.

### Requirements
For this please follow the below steps:
1. Create a virtual environment to do install the libraries; follow this [link](https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/) in case you don't know how to do it; you possibly need to install [pip](https://www.makeuseof.com/tag/install-pip-for-python/), if you don't have it installed
2. Run the following command: 
```shell
pip install -r requirements.txt
```
This will install all of the libraries references in the requirements.txt file.
1. When you have finished using the package or working on your project, you can deactivate the virtual environment:
```shell
deactivate
```
This command exits the virtual environment and returns you to your normal command prompt.
1. Enjoy! :)

### Package Distributions
1. Locate the `dist` folder in your project's root directory. This folder contains the package distributions, including the source distribution (`*.tar.gz` file) and the wheel distribution (`*.whl` file).

2. Install the package using one of the following methods:

   - Install the source distribution:
   ```shell
   pip install dist/msecnn_raulkviana-1.0.tar.gz
   ```

   - Install the wheel distribution:
   ```shell
   pip install dist/msecnn_raulkviana-1.0.whl
   ```

3. Once the package is installed, you can import and use its functionalities in your Python code.

## 7. Demo


## 8. Contributions

Feel free to contact me through this [email](raulviana@ua.pt) or create either a issue or pull request to contribute to this project ^^.

## 9. License

This project license is under the [MIT License](LICENSE).

## 10. TODO

|Task| Description| Status (d - doing, w - waiting, f- finished)|
|-----|-----|-----|
| Implement code to test functions| Use a library, such as Pytest, to test some functions from the many modules developed | w |

## 11. References
[1] T. Li, M. Xu, R. Tang, Y. Chen, and Q. Xing, [“DeepQTMT: A Deep Learning Approach for Fast QTMT-Based CU Partition of Intra-Mode VVC,”](https://arxiv.org/abs/2006.13125) IEEE Transactions on Image Processing, vol. 30, pp. 5377–5390, 2021, doi: 10.1109/tip.2021.3083447.
[2] R. K. Viana, “Deep learning architecture for fast intra-mode CUs partitioning in VVC,” Universidade de Aveiro, Nov. 2022.

<div align="center">
  <img src="../../img/funny_memes_about_this_work/72rukf.gif" width=450 />
  <p>:)</p>
</div>




With this demo you will be able to understand better how the MSE-CNN works and its goal! :D

<center>
<img src="file/msecnn_model.png" width=500 />
</center>

## Tutorial
To use this demo follow these steps ;):
1. Load an image from your PC. Since the model needs to be fed 128x128 images, if your image is larger than that, then the 128 by 128 area within your image top left will be loaded. Additionally, you can tell the app the coordinates of the area of the snapshot you want to see. The coordinates are related to the image's upper-left corner, therefore position 0,0 refers to that particular area. Furthermore, you must first specify the height position (y axis) and then the width position (x axis).
2. Click "Submit" to pass the image through the model. 
3. After the previous step, an image will be displayed with the best way, according with the model, to partition that section of the image. The possible ways to partition an image in VVC is in Quartenary Tree (QT), Binary Tree Horizontal or Vertical (BTH or BTV), Ternary Tree Horizontal or Vertical (TTH, TTV) and Non-split.

**Note**: This demo implementation has some limitations, such as the fact that the model occasionally makes illogical predictions. For instance, splitting a 16x32 CU vertical rectangle using VTT is incorrect. This occurs as a result of the model's inherent constraints as well as the fact that only the best split is being chosen. One way to reduce this behaviour is to evaluate not only the optimal split but also alternate splits. This can be accomplished by, for instance, applying the multi-thresholding method to the model's forecasts and determining the splits that are most likely to occur. Additionally, when unreasonable splits are predicted, the code immediately halts the partitioning of that particular block.

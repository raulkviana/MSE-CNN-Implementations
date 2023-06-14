With this demo you will be able to understand better how the MSE-CNN works and its goal!

<center>
<img src="file/msecnn_model.png" width=500 />
</center>

## Tutorial
To use this demo follow these steps:
1. Load an image from your PC. Since the model needs to be fed 128x128 images, if your image is larger than that then the 128 by 128 area within your image top left will be loaded.
2. Click "Compute" to pass the image through the model. 
3. After the previous step and image will be displayed with the best way, according with the model, to partition that section of the image. The possible ways to partition an image in VVC is in Quartenary Tree (QT), Binary Tree Horizontal or Vertical (BTH or BTV), Ternary Tree Horizontal or Vertical (TTH, TTV) and Non-split (which mean that the section should not be split)
4. 
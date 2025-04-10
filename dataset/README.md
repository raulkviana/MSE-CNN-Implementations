# Data

This folder contains all of the data used in this work. To train a supervised ML model, annotated data must be created. Given that a trained network has inputs and outputs with specifications, the data supplied to the model must be modified to meet those. This modification entails certain data processing steps to generate data that can directly aid network training/evaluation. A five-step method was designed to process the data, as shown in the diagram below. 

![labels_gen](https://github.com/raulkviana/MSE-CNN-Implementations/blob/assets/labels_gen_method.png)

You can learn more about the code inside each step from the above image in [processing_data_code.md](processing_data_code.md)

## Available data

| Type of data |Description| 
|--------------|-----------|
| Images | Images from gathered by Tianyi Li et al that can be used to obtain labels.  |
| Videos | Videos from gathered by Tianyi Li et al that can be used to obtain labels. |
| [Entire Encoded Images with a QP of 32](https://drive.google.com/drive/folders/1MpoJs8fWMQof_Q8RFLxAvGTXlequ187I?usp=sharing) | All RAISE images encoded using the encoder [VTM-7.0](https://github.com/tianyili2017/CPIV/blob/master/VTM-7.0_Data.zip). This tool encodes the database and determines the best method for partitioning the sequences contained within it, while also indicating the RD cost for each of these partitions. |
| [Structured data](https://drive.google.com/drive/folders/10wC0sfyYOPw85kaJJeK91KGAHW8Qf415?usp=sharing)| Because the encoded data consists of a large number of files containing information about specific CUs with no correlation, the data must be structured. This structure contains information about the CU's location, size, file name, picture order count (POC), and optimal split mode. The data is in the luma channel. |
| Data with real CTUs | Because the MSE-input CNN's is a CTU, this information must be added to the labels. A new property called "real CTU" is added to the sequence structure, which includes the actual CTU splitting structure from the images. The data is in the luma channel. |
| Filtered data for training each stage | It is proposed in the original paper that this network be trained stage by stage and with specific CU types. This means that not all CUs will be required at the same time during training. To organise the data in this manner, a search of the previous step's data is performed, and changes are made to the data structure containing the sequences. The data is in the luma channel.|
| Balanced dataset | The data up to this point is imbalanced and not suitable for training because there are different numbers of split modes. The mode may not be able to learn how to predict underrepresented partitions if this data is fed to it. Because of this, the network assumes that each class will be the one with the highest representation in the dataset. To balance the data, the sample size has been reduced. The data is in the luma channel.|

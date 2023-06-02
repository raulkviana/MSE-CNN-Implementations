# Data

This folder contains all of the data used in this work. To train a supervised ML model, annotated data must be created. Given that a trained network has inputs and outputs with specifications, the data supplied to the model must be modified to meet those. This modification entails certain data processing steps to generate data that can directly aid network training/evaluation. A five-step method was designed to process the data, as shown in the diagram below. 

![labels_gen](../../imgs/labels_gen_method.png)

You can learn more about the code inside each step from the above image in [processing_data_code.md](processing_data_code.md)

## Available data

| Type of data |Description| 
|--------------|-----------|
| [Images](https://uapt33090-my.sharepoint.com/:f:/g/personal/raulviana_ua_pt/ErGMPFUtH7BMkdNu3KiTz1sBcqiW78JDAcoymWU5HKgbug?e=fKNfdy) | Images from gathered by Tianyi Li et al that can be used to obtain labels.  |
| [Videos](https://uapt33090-my.sharepoint.com/:f:/g/personal/raulviana_ua_pt/EmmY3sQqrcBGuPyIoWncQdkB8DpEqZAqCsO0kIR8_9nuEw?e=vnve6T) | Videos from gathered by Tianyi Li et al that can be used to obtain labels. |
| [Entire Encoded Images with a QP of 32](https://uapt33090-my.sharepoint.com/:f:/g/personal/raulviana_ua_pt/Eo0u_LRfxlNDqdK2Q4rPXfkBI_FQzUMW2DiJpkfpTy9kZQ?e=DYmam3) | All RAISE images encoded using the encoder [VTM-7.0](https://github.com/tianyili2017/CPIV/blob/master/VTM-7.0_Data.zip). This tool encodes the database and determines the best method for partitioning the sequences contained within it, while also indicating the RD cost for each of these partitions. |
| [Structured data](https://uapt33090-my.sharepoint.com/:f:/g/personal/raulviana_ua_pt/EpF90NJ8QNdGhpc5rjJfQygBrQ9GD8D77gDAXiib4mDZiw?e=LNXU2L)| Because the encoded data consists of a large number of files containing information about specific CUs with no correlation, the data must be structured. This structure contains information about the CU's location, size, file name, picture order count (POC), and optimal split mode. The data is in the luma channel. |
| Data with real CTUs | Because the MSE-input CNN's is a CTU, this information must be added to the labels. A new property called "real CTU" is added to the sequence structure, which includes the actual CTU splitting structure from the images. The data is in the luma channel. |
| Filtered data for training each stage | It is proposed in the original paper that this network be trained stage by stage and with specific CU types. This means that not all CUs will be required at the same time during training. To organise the data in this manner, a search of the previous step's data is performed, and changes are made to the data structure containing the sequences. The data is in the luma channel.|
| Balanced dataset | The data up to this point is imbalanced and not suitable for training because there are different numbers of split modes. The mode may not be able to learn how to predict underrepresented partitions if this data is fed to it. Because of this, the network assumes that each class will be the one with the highest representation in the dataset. To balance the data, the sample size has been reduced. The data is in the luma channel.|

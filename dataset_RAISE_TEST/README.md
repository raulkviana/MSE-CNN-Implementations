# Data: RAISE_TEST

This folder contains all of the data used in this work, more especifically the RAISE_TEST files from the original database. To train a supervised ML model, annotated data must be created. Given that a trained network has inputs and outputs with specifications, the data supplied to the model must be modified to meet those. This modification entails certain data processing steps to generate data that can directly aid network training/evaluation. A five-step method was designed to process the data, as shown in the diagram below. 

![labels_gen](../imgs/labels_gen_method.png)

A table with links to this data is shown below. 

| Type of data |Description| 
|--------------|-----------|
| [Encoded data](https://uapt33090-my.sharepoint.com/:f:/g/personal/raulviana_ua_pt/ElxwuCMypJJGlNLhofrzqmABjAuYHZCXATWN259iEZBiig?e=jMRPw4) | RAISE_TEST files encoded using the encoder [VTM-7.0](https://github.com/tianyili2017/CPIV/blob/master/VTM-7.0_Data.zip). This tool encodes the database and determines the best method for partitioning the sequences contained within it, while also indicating the RD cost for each of these partitions. |
| [Structured data](https://uapt33090-my.sharepoint.com/:f:/g/personal/raulviana_ua_pt/EpF90NJ8QNdGhpc5rjJfQygBrQ9GD8D77gDAXiib4mDZiw?e=LNXU2L)| Because the encoded data consists of a large number of files containing information about specific CUs with no correlation, the data must be structured.  This structure contains information about the CU's location, size, file name, picture order count (POC), and optimal split mode. |
| [Data with real CTUs](https://uapt33090-my.sharepoint.com/:f:/g/personal/raulviana_ua_pt/Eol2hN8Eze1OoiJSFOIC3XgB6WyFFz-OchsIBZCP0JtVFg?e=0vxO5V) | Because the MSE-input CNN's is a CTU, this information must be added to the labels. A new property called "real CTU" is added to the sequence structure, which includes the actual CTU splitting structure from the images. |
| [Filtered data for training each stage ]( )| It is proposed in the original paper that this network be trained stage by stage and with specific CU types. This means that not all CUs will be required at the same time during training. To organise the data in this manner, a search of the previous step's data is performed, and changes are made to the data structure containing the sequences. |
| [Balanced dataset](https://uapt33090-my.sharepoint.com/:f:/g/personal/raulviana_ua_pt/EjypictyMVBKmq1ZpKZkL9UBdy04NI-uxACygvxewr4VYQ?e=nL82lA) | The data up to this point is imbalanced and not suitable for training because there are different numbers of split modes. The mode may not be able to learn how to predict underrepresented partitions if this data is fed to it. Because of this, the network assumes that each class will be the one with the highest representation in the dataset. To balance the data, the sample size has been reduced. |

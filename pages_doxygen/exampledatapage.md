# Example data

The data in this folder is used by the scripts in the src/msecnn_raulkviana/usefull_scripts folder. The data in this folder is unbalanced. To load this data into the custom data class, you can follow this sample of code

```python
import custom_dataset

# Path to already processed data
l_path = "path_to_label"

# Instatiate dataset class to train stage 6 (there is also classes to train other stages)
dataset = custom_dataset.CUDatasetStg6(files_path=l_path_train)  # Dataset for training of stage 6
batch_sampler = custom_dataset.SamplerStg6(train_data, batch_size)  # Batch Sampler to batch cus of the same size together
dataloader_train = DataLoader(train_data, num_workers=num_workers, batch_sampler=batch_sampler_train)  # Data Loader that can be used to train or evaluate the model 
```


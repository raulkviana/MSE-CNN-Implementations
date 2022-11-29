import dataset_utils

def main():

    # Directory containing the .txt files with CUs informations
    d_path = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/valid/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_64x64_v2/complexity"
    print("Dir path:", d_path)

    dataset_utils.gen_dataset_types(d_path, 0.1)

if __name__ == "__main__":
    main()

import dataset_utils
import time

def main():

    # Directory containing the .txt files with CUs informations
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_32x32_v2/train_valid_test"  # Path with the labels processed
    #path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels_for_testing/train/mod_with_struct_change_32x32"
    path_dir_l = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/test/processed_labels/mod_with_real_CTU/mod_with_struct_change_no_dupl_stg6_v4/train_valid_test/balanced_labels_downsamp/train_val"

    print("Balance data in:", path_dir_l)

    # start = time.time()
    # dataset_utils.balance_dataset_JF(path_dir_l, n_classes=6)  # Upsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_JF:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_down(path_dir_l, n_classes=2)  # Downsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_down:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_up(path_dir_l, n_classes=6)  # Upsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_up:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_down_v2(path_dir_l)  # Downsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_down_v2:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_down_v4(path_dir_l)  # Downsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_down_v3:", end-start)

    start = time.time()
    pos_to_find_split = -3
    dataset_utils.get_some_data_equaly(100000, path_dir_l, 2, pos_to_find_split)  # Downsampling
    end = time.time()
    print("Elapsed time for get_some_data_equaly:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_up_v2(path_dir_l)  # Upsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_up_v2:", end-start)

    # start = time.time()
    # dataset_utils.balance_dataset_up_v3(path_dir_l)  # Upsampling
    # end = time.time()
    # print("Elapsed time for balance_dataset_up_v3:", end-start)

if __name__ == "__main__":
    main()

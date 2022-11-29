import dataset_utils


def main():

    # Directory containing the .dat files with CUs informations
    #d_path = str(input("Em que path absoluto está o dataset?\n"))
    #d_path = '/mnt/c/Users/Raul/Dropbox/Dataset/Img/Example'
    #d_path = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/labels/train"
    d_path = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/data_for_now/valid/"

    #dataset_utils.unite_labels_v6(dir_path_l=d_path, n_output_file="labels_pickkklke")  # The result is saved in the same folder
    dataset_utils.unite_labels_v6_mod(dir_path_l=d_path, n_output_file="processed_labels")  # The result is saved in the same folder

if __name__ == "__main__":
    main()

import CustomDataset
import dataset_utils

def main():

    # Directory containing the .txt files with CUs informations
    f_path = "/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/data/RAISE_Test_768x512.yuv"
    file_info = dataset_utils.get_file_metadata_info("/nfs/home/rviana.it/MSE_CNN/Dataset_Labels/all_data/data/", "RAISE_Test_768x512.yuv")
    f_size = (file_info["height"], file_info["width"])
    cu_pos = (0, 0)
    cu_size = (128, 128)
    frame_number = 0
    print("Path:", f_path)
    yuv_frame, CU_Y, CU_U, CU_V =CustomDataset.get_cu(f_path, f_size, cu_pos, cu_size, frame_number)

if __name__ == "__main__":
    main()

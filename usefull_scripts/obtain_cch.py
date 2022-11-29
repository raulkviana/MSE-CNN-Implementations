import dataset_utils

def main():

    # Directory containing the .txt files with CUs informations
    #d_path = str(input("Em que path absoluto está o dataset?\n"))
    #d_path = '/mnt/c/Users/Raul/Dropbox/Dataset/Img/Example'
    d_path = r"C:\Users\Raul\Dropbox\Dataset\Img\Test_place\labels_pickle"

    dataset_utils.labels_with_specific_cch(dir_path=d_path, cch=0)
    dataset_utils.labels_with_specific_cch(dir_path=d_path, cch=1)

if __name__ == "__main__":
    main()

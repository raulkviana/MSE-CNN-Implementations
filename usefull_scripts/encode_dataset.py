import dataset_utils

def main():

    # Quantization parameter
    #QP = str(input("Qual o valor para o quantization parameter (QP)?\n"))
    QP = 27

    # Temporal Subsample Ratio
    #ts = str(input("Qual o valor para o Temporal Subsample Ratio (ts)?\n"))
    ts = 500

    # Directory containing the dataset images
    #d_path = str(input("Em que path absoluto está o dataset?\n"))
    d_path = '/mnt/c/Users/Raul/Dropbox/Dataset/Img/Example'
    d_path = "C:/Users/Raul/Dropbox/Dataset/Img/Example"

    # Directory containing the .exe file extension to run the encoder.
    # It ends with CPIV\VTM-7.0_Data\bin\vs16\msvc-19.24\x86_64\release in Windows
    e_path = "C:/Users/Raul/Documents/GitHub/CPIV/VTM-7.0_Data/bin/vs16/msvc-19.24/x86_64/release" # Example for windows
    #e_path = "/mnt/c/Users/Raul/Documents/GitHub/CPIV/VTM-7.0_Data/bin/" # Example for linux
    #e_path = str(input("Em que path absoluto está o encoder?\n"))

    dataset_utils.encode_dataset(d_path=d_path, e_path=e_path, ts=ts, QP=QP) # The result is saved in the encoder folder

if __name__ == "__main__":
    main()

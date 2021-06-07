import os



def load_dataset_filename(img_folder_path):
    os.chdir("/gdrive/MyDrive/ISBI_Dataset")
    X_ids = next(os.walk('train'))[2]
    Y_ids = next(os.walk('label'))[2]
    print(len(X_ids),len(Y_ids))
    X_ids.sort()
    Y_ids.sort()
    return X_ids,Y_ids


def 



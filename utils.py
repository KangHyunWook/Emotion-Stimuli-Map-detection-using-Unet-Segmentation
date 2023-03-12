from skimage import transform
from keras import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import numpy as np

def splitData(img_folder, mask_folder):

    filePaths = getImagePaths(img_folder)
    maskPaths = getImagePaths(mask_folder)

    state=7
    filePaths = shuffle(filePaths, random_state = state)
    maskPaths = shuffle(maskPaths, random_state = state)

    n_train = int(len(filePaths)*0.9)

    trainFile_paths = filePaths[:n_train]
    mask_trainFile_paths = maskPaths[:n_train]

    testFile_paths = filePaths[n_train:]
    mask_testFile_paths = maskPaths[n_train:]

    train_f = open('./train_files.txt', 'w')
    test_f = open('./test_files.txt', 'w')

    for i in range(len(trainFile_paths)):
        train_f.write(trainFile_paths[i]+'\n')
        train_f.write(mask_trainFile_paths[i]+'\n')

    for i in range(len(testFile_paths)):
        test_f.write(testFile_paths[i]+'\n')
        test_f.write(mask_testFile_paths[i]+'\n')

    train_f.close()
    test_f.close()

def getImagePaths(root):
    items = os.listdir(root)
    pathList=[]
    for item in items:
        full_path = os.path.join(root, item)
        if os.path.isfile(full_path):
            pathList.append(full_path)
        else:
            pathList.extend(getImagePaths(full_path))
    return pathList

def preprocess_img(file_path):
    x=preprocessing.image.load_img(file_path, grayscale=True)
    x = preprocessing.image.img_to_array(x)
    x = transform.resize(x, (128,128,1), mode='constant', preserve_range = True)
    x /=255.

    return x

def mapImageLabel(img_file_name):

    img_list=[]
    mask_list=[]
    f= open(img_file_name)
    for file_path in f:
        file_path=file_path.strip()

        splits=file_path.split(os.path.sep)

        data_type = splits[-3]

        img = preprocess_img(file_path)
        if data_type == 'images':
            img_list.append(img)
        elif data_type =='ground_truth':
            mask_list.append(img)

    X=np.asarray(img_list)
    y=np.asarray(mask_list)

    return X, y

def split_train_test(train_file_name,test_file_name):

    train_X, train_y = mapImageLabel(train_file_name)

    test_X, test_y = mapImageLabel(test_file_name)

    return train_X, test_X, train_y, test_y





#

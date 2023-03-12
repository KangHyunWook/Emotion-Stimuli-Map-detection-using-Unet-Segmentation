from skimage import transform
from keras import preprocessing
from sklearn.model_selection import train_test_split
import os
import numpy as np

def getImagePaths(root):
    items= os.listdir(root)
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

def preprocessData(args):
    imgPathList=getImagePaths(args.img_root)
    maskPathList=getImagePaths(args.mask_root)

    X=[]
    y=[]

    for i in range(len(imgPathList)):
        #todo: call preprocess_img funcition
        x_img = preprocess_img(imgPathList[i])
        X.append(x_img)

        ground_truth_img = preprocess_img(maskPathList[i])
        y.append(ground_truth_img)

    X=np.asarray(X)
    y=np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

import numpy as np
import cv2
import os
data_root = '/database/daehyeon/bu101/images/'
folder_list = os.listdir(data_root) ## swing,golf,....
for i,folder_name in enumerate(folder_list):
    file_list = os.listdir(os.path.join(data_root,folder_name))
    for j,image in enumerate(file_list):
        full_path = os.path.join(data_root,folder_name,image)
        if i==0 and j==0:
            img = cv2.imread(full_path)
            img = cv2.resize(img,(224,224))
            img = np.expand_dims(img,axis=0)
        else:
            img_ = cv2.imread(full_path)
            img_ = cv2.resize(img_,(224,224))
            img_ = np.expand_dims(img_,axis=0)
            img = np.concatenate((img, img_), axis=0)
    print(i,'/',img.shape)

np.save('/database/daehyeon/bu101/data.npy',img)
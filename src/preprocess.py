import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from torchvision import transforms, utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import torch

#MASK_DIR = "../data/masks"
#IMG_DIR = "../data/images"
IMG_DIR = "../data/test_images"
MASK_DIR = "../data/test_masks"

image_file_names = [f for f in listdir(IMG_DIR) if isfile(join(IMG_DIR, f))]
mask_file_names = [f for f in listdir(MASK_DIR) if isfile(join(MASK_DIR, f))]

def tensorize_image(image_path, output_shape):
    batch_images = []
    for file_name in image_path:
        os_image_path = os.path.join(IMG_DIR, file_name)
        img = cv2.imread(os_image_path)

        copy_img = img.copy() 
        copy_img = cv2.resize(img,output_shape)
        batch_images.append(copy_img)

        
    batch_images = np.array(batch_images)
    image_tensor = torch.Tensor(batch_images)
    #[4765, 20, 20, 3] [B,W,H,C]
    print(image_tensor.shape)


def tensorize_mask(mask_path, output_shape):
    batch_masks = list()
    for file_name in mask_path:
        os_mask_path = os.path.join(MASK_DIR, file_name)
        mask = cv2.imread(os_mask_path, cv2.IMREAD_GRAYSCALE)

        copy_mask = mask.copy() 
        copy_mask = cv2.resize(mask, output_shape)


        # ?? emin değilim
        #encoder = OneHotEncoder()
        #encoder.fit(copy_mask)
        #encoded = encoder.fit_transform(copy_mask)
        #print(encoded.size)
        #print("encoded", encoded.shape)

        copy_mask = np.asarray(copy_mask) / 255

        batch_masks.append(copy_mask)

    #batch_masks = np.vstack(batch_masks)   
    batch_masks = np.array(batch_masks)
    mask_tensor = torch.Tensor(batch_masks)
    mask_tensor = mask_tensor.unsqueeze(3)
    #[4765, 20, 20] [B,W,H] channel sayısı = 2 gösteremedim
    print(mask_tensor.shape) 

tensorize_image(image_file_names,(20,20))
print("------------")    
tensorize_mask(mask_file_names ,(20,20))
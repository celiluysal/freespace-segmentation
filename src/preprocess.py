import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from torchvision import transforms, utils
from sklearn.preprocessing import OneHotEncoder

MASK_DIR = "../data/masks"
IMG_DIR = "../data/images"
IMG_DIR_TEST = "../data/test_images"
MASK_DIR_TEST = "../data/test_masks"

image_file_names = [f for f in listdir(IMG_DIR_TEST) if isfile(join(IMG_DIR_TEST, f))]
mask_file_names = [f for f in listdir(MASK_DIR_TEST) if isfile(join(MASK_DIR_TEST, f))]

transform_toPILimage = transforms.ToPILImage()
transform_toTensor = transforms.ToTensor()


def tensorize_image(image_path, output_shape):
    batch_images = []
    for file_name in image_path:
        os_image_path = os.path.join(IMG_DIR_TEST, file_name)
        img = cv2.imread(os_image_path)
        #print(img.shape)
        copy_img = img.copy() 
        copy_img = cv2.resize(img,output_shape)
        #print(copy_img.shape)
        PIL_image = transform_toPILimage(copy_img)
        image_tensor = transform_toTensor(PIL_image)
        batch_images.append(image_tensor)
        print(image_tensor.shape)
        
        #atch_images.append(copy_img)
        #print(batch_images)
    


def tensorize_mask(mask_path, output_shape):
    for file_name in mask_path:
        os_mask_path = os.path.join(MASK_DIR_TEST, file_name)
        mask = cv2.imread(os_mask_path, cv2.IMREAD_GRAYSCALE)
        #print(mask.shape)
        copy_mask = mask.copy() 
        copy_mask = cv2.resize(mask,output_shape)

        encoder = OneHotEncoder()
        encoder.fit(copy_mask)
        mask_tensor = transform_toTensor(copy_mask)

        #print(mask_tensor)

        print(mask_tensor.shape)  
        #print(type(mask_tensor))



tensorize_image(image_file_names,(200,200))
print("------------")
    
tensorize_mask(mask_file_names ,(200,200))
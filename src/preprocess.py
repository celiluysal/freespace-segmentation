import numpy as np
import cv2
import torch
import glob

# MASK_DIR = "../data/masks"
# IMG_DIR = "../data/images"
MASK_DIR = "../data/test_masks"
IMG_DIR = "../data/test_images"

def tensorize_image(image_path, output_shape):
    batch_images = []
    for file_name in image_path:
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        img = cv2.resize(img,output_shape)
        batch_images.append(img)

    batch_images = np.array(batch_images)
    torch_image = torch.from_numpy(batch_images)
    image_tensor = torch_image.cuda()
    return image_tensor 
    #[4765, 20, 20, 3] [B,W,H,C]
    

def tensorize_mask(mask_path, output_shape ,n_class):
    batch_masks = list()
    for file_name in mask_path:
        mask = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, output_shape)
        mask = mask / 255
        print(mask)
        mask = one_hot_encode(mask, n_class)  
        batch_masks.append(mask)      
  
    batch_masks = np.array(batch_masks)
    # print("mask enc batch", batch_masks)
    torch_mask = torch.from_numpy(batch_masks)
    # print("mask torch", torch_mask)
    mask_tensor = torch_mask.cuda()
    # print("mask tens", mask_tensor)
    # print("mask tens len", len(mask_tensor))
    #[4765, 20, 20, 2] [B,W,H,C] 
    return mask_tensor

def one_hot_encode(data, n_class):
    encoded_data = np.zeros((data.shape[0], data.shape[1], n_class), dtype=np.int)
    encoded_labels = []
    for lbl in range(n_class):
        encoded_label = [0] * n_class 
        encoded_label[lbl] = 1
        encoded_labels.append(encoded_label)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if ((data[i][j] == 0).all()):
                encoded_data[i, j] = encoded_labels[0]
            elif ((data[i][j] == 1).all()):
                encoded_data[i, j] = encoded_labels[1]
    print(encoded_data)
    return encoded_data


if __name__ == '__main__':
    
    # image_file_names = glob.glob(IMG_DIR + "/*")
    # image_file_names.sort()
    # batch_image_list = image_file_names[:5] #first n
    # batch_image_tensor = tensorize_image(batch_image_list, (20,20))
    
    # print(batch_image_list)
    # print(batch_image_tensor.dtype)
    # print(type(batch_image_tensor))
    # print(batch_image_tensor.shape)

    # print("------------")    
    
    mask_file_names = glob.glob(MASK_DIR + "/*")
    mask_file_names.sort()
    batch_mask_list = mask_file_names[:1] #first n
    batch_mask_tensor = tensorize_mask(batch_mask_list, (5,5), 2)
    
    print(batch_mask_list)
    print(batch_mask_tensor.dtype)
    print(type(batch_mask_tensor))
    print(batch_mask_tensor.shape)  

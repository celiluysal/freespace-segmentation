from model import FoInternNet
# from unet import UNet

from preprocess import tensorize_image,tensorize_mask, image_mask_check, decode_and_convert_image
import os
import glob, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 4
epochs = 1
cuda = True
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
# IMAGE_DIR = os.path.join(DATA_DIR, 'test_images')
# MASK_DIR = os.path.join(DATA_DIR, 'test_masks')
###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=2)
# model = UNet(n_channels=3, n_classes=2, bilinear=True)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

# TRAINING THE NEURAL NETWORK
outputs_list1 = []
outputs_list2 = []

for epoch in range(epochs):
    running_loss = 0
    # print("epoch ", epoch)
    # print("steps_per_epoch: {}",steps_per_epoch)
    for ind in tqdm.tqdm(range(steps_per_epoch)):
        #472-476
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        # batch_input = batch_input.to('cuda')


        # print(batch_input_path_list)
        # print("image shape: ", batch_input.shape)
        # print(batch_input)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        # batch_label = batch_label.to('cuda')
        # print(batch_label_path_list)
        # print("mask shape : ", batch_label.shape)

        optimizer.zero_grad()

        # print(batch_label.shape)
        # print(batch_label)
        outputs = model(batch_input)
        test_output1 = outputs
        # print("output1 ", test_output1.shape)
        # if(epoch == 1):
            # outputs_list1.append(outputs)
        # print(outputs[0])
        # print("2------------------------")
        # print("outputs: ",outputs)
        loss = criterion(outputs, batch_label)
        # print("2------------------------")
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(ind)
        if ind == steps_per_epoch-1:
            str1 = 'training loss on epoch   {}: {}'.format(epoch, running_loss)
            print(str1)

            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                test_output2 = outputs
                # print("output2 ", test_output2.shape)
                # if(epoch == 1):
                    # outputs_list2.append(outputs)

                loss = criterion(outputs, batch_label)
                val_loss += loss
                break
            
            str2 = 'validation loss on epoch {}: {}'.format(epoch, val_loss)
            print(str2)

    
    if(epoch == epochs-1):
        torch.cuda.empty_cache()
        test_output_list = []
        for test_input_path, test_label_path in zip(test_input_path_list, test_label_path_list):
            batch_input = tensorize_image([test_input_path], input_shape, cuda)
            # batch_label = tensorize_mask([test_label_path], input_shape, n_classes, cuda)
            # print("1----")
            outputs = model(batch_input)
            # print("2----")
            test_output_list.append(outputs)


# img_list = decode_and_convert_image(outputs, n_classes)
# img = img_list[0]

print(len(test_output_list))
print(test_output_list[0])

# print(test_output1.shape)
# print(test_output1)
# print(test_output2.shape)
# print(test_output2)
# print(img.shape)
# print(img)
# plt.imshow(img, cmap="gray")
# plt.show()


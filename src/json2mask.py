import numpy as np
import cv2
import json
import os
import tqdm
import re

MASK_DIR  = '../data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

JSON_DIR = '../data/jsons'

# def clean_TypeName(_name):
#     pattern = r"\.[0-9a-z].*"
#     name = re.sub(pattern, '', _name)
#     return name

json_list = os.listdir(JSON_DIR)
json_list.sort()

for json_name in tqdm.tqdm(json_list):
    # file_name = clean_TypeName(json_name)
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')
    json_dict = json.load(json_file)

    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    
    mask_path = os.path.join(MASK_DIR, json_name[:-5])
    
    for obj in json_dict["objects"]:
        if obj['classTitle']=='Freespace':
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)
            
    cv2.imwrite(mask_path, mask.astype(np.uint8))
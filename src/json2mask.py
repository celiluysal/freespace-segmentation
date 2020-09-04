from os import listdir
from os.path import isfile, join
import numpy as np
import json
import os
import cv2
        
json_dicts = []
JSON_DIR = "../data/jsons"
MASK_DIR = "../data/masks"

file_names = [f for f in listdir(JSON_DIR) if isfile(join(JSON_DIR, f))]


for name in file_names:
    json_name = name
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')
    json_dict = json.load(json_file)
    json_dicts.append(json_dict)
    
json_objs = json_dict["objects"]

for obj in json_objs:
    if obj["classTitle"] == "Freespace":
        points = obj["points"]["exterior"]
        print(points)

## draw filled polygon width, height, points 
mask = np.zeros((1208, 1920, 3), np.uint8)
pts = np.array(points, np.int32)
pts = pts.reshape((-1,1,2))

cv2.fillPoly(mask,[pts],(255,255,255))
cv2.imshow('Window', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(join(MASK_DIR,"asd.png"),mask)
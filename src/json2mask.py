from os import listdir
from os.path import isfile, join
import numpy as np
import json
import os
import cv2


        
json_dicts = []
JSON_DIR  = '../data/jsons'
file_names = [f for f in listdir(JSON_DIR) if isfile(join(JSON_DIR, f))]


for name in file_names:
    json_name = name
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')
    json_dict = json.load(json_file)
    json_dicts.append(json_dict)
    
json_objs = json_dict["objects"]

for obj in json_objs:
    if obj["classId"] == 38:
        points = obj["points"]
        exterior = points["exterior"]
        print(exterior)

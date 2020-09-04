from os import listdir
from os.path import isfile, join
import numpy as np
import json
import os
import cv2
        
images = []
JSON_DIR = "../data/jsons"
MASK_DIR = "../data/masks"

file_names = [f for f in listdir(JSON_DIR) if isfile(join(JSON_DIR, f))]

class image:
    def __init__(self, width, height, points, name):
        self.width = width
        self.height = height
        self.points = points
        self.name = name

for file_name in file_names:
    json_path = os.path.join(JSON_DIR, file_name)
    json_file = open(json_path, 'r')
    json_dict = json.load(json_file)
    width = json_dict["size"]["width"]
    height = json_dict["size"]["height"]
    json_objs = json_dict["objects"]
    
    for obj in json_objs:
        if obj["classTitle"] == "Freespace":
            points = obj["points"]["exterior"]
            img = image(width, height, points, file_name)
            images.append(img)

for image in images:
    draw_and_save_filledPolygon(image.width, image.height, image.points, image.name)

## draw filled polygon width, height, points, name
def draw_and_save_filledPolygon(_width, _height, _points, _name):
    mask = np.zeros((_width, _height, 3), np.uint8)
    pts = np.array(_points, np.int32)
    pts = pts.reshape((-1,1,2))
    
    cv2.fillPoly(mask,[pts],(255,255,255))
    cv2.imwrite(join(MASK_DIR,_name+".png"),mask)
from os import listdir
from os.path import isfile, join
import numpy as np
import json
import os
import cv2
import re
        
image_mask_list = []
image_list = []

JSON_DIR = "../data/jsons"
MASK_DIR = "../data/masks"
IMG_DIR = "../data/images"
MASK_IMG_DIR = "../data/masked_images"

json_file_names = [f for f in listdir(JSON_DIR) if isfile(join(JSON_DIR, f))]
image_file_names = [f for f in listdir(IMG_DIR) if isfile(join(IMG_DIR, f))]

print("seea")


def clean_TypeName(_name):
        pattern = r'\..*'
        name = re.sub(pattern, '', _name)
        return name

class ImageMask(object):
    def __init__(self, label=None):
        self.label = label
        self.parent = None
        self.children = []
    def __init__(self, width, height, points, name):
        self.width = width
        self.height = height
        self.points = []
        self.points.append(points)
        self.name = clean_TypeName(name)
        
    
def read_jsons():
    for file_name in json_file_names:
        json_path = os.path.join(JSON_DIR, file_name)
        json_file = open(json_path, 'r')
        json_dict = json.load(json_file)
        width = json_dict["size"]["width"]
        height = json_dict["size"]["height"]
        json_objs = json_dict["objects"]
        fs_count = 0
        
        for obj in json_objs:
            if obj["classTitle"] == "Freespace":
                fs_count += 1
                points = obj["points"]["exterior"]
                img = ImageMask(width, height, points, file_name)
                if(fs_count > 1):
                    print(img.name)
                    img.points.append(points)
        image_mask_list.append(img)

            
def draw_and_save_filledPolygon(_ImageMask:ImageMask):
    mask = np.zeros((_ImageMask.width, _ImageMask.height, 3), np.uint8)
    
    points_list = _ImageMask.points
    for points in points_list:
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(mask,[pts],(255,255,255))
    
    cv2.imwrite(join(MASK_DIR,_ImageMask.name+"_mask.png"),mask)
    
      
    
def draw_and_save_filledPolygon_onImage(_ImageMask:ImageMask, _file_name):
    image_path = os.path.join(IMG_DIR, _file_name)
    png = cv2.imread(image_path)
    copy_png = png.copy()
    
    points_list = _ImageMask.points
    for points in points_list:
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1,1,2))
        green = (153,255,51)
        purple = (204,0,255)
        cv2.fillPoly(copy_png,[pts],green)
        opacity = 0.3
        cv2.addWeighted(copy_png,opacity,png,1-opacity,0,png)
    cv2.imwrite(join(MASK_IMG_DIR,_ImageMask.name+"masked_image.png"),png)


            
#read_jsons()
#for ImageMask in image_mask_list:
#    draw_and_save_filledPolygon(ImageMask)

#image_file_names.sort()
#for file_name in image_file_names:
#    for image_mask in image_mask_list:
#        name = clean_TypeName(file_name)
#        if(image_mask.name == name):
#            draw_and_save_filledPolygon_onImage(image_mask, file_name)
#            break
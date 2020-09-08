# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:41:54 2020

@author: celil
"""


from os import listdir
from os.path import isfile, join
import numpy as np
import json
import os
import cv2
        
images = []
JSON_DIR = "../data/jsons"
MASK_DIR = "../data/masks"
# file_names = [f for f in listdir(JSON_DIR) if isfile(join(JSON_DIR, f))]

class image:
    def __init__(self, width, height, points, name):
        self.width = width
        self.height = height
        self.points = []
        self.points.append(points)
        self.name = name
        
im1 = image(200, 200, [[0,0],[50,0],[50,50],[0,50],[0,0]],"image1")
im1.points.append([[200,0],[200,50],[150,50],[150,0]])
images.append(im1)

#for file_name in file_names:
#    json_path = os.path.join(JSON_DIR, file_name)
#    json_file = open(json_path, 'r')
#    json_dict = json.load(json_file)
#    width = json_dict["size"]["width"]
#    height = json_dict["size"]["height"]
#    json_objs = json_dict["objects"]
#    fs_count = 0
#    
#    for obj in json_objs:
#        if obj["classTitle"] == "Freespace":
#            fs_count += 1
#            points = obj["points"]["exterior"]
#            img = image(width, height, points, file_name)
#            images.append(img)
#            if(fs_count > 1):
#                print(file_name)


            
## draw filled polygon width, height, points, name
def draw_and_save_filledPolygon(_width, _height, _points, _name):
    mask = np.zeros((_width, _height, 3), np.uint8)
    pts = np.array(_points, np.int32)
    pts = pts.reshape((-1,1,2))
    
    cv2.fillPoly(mask,[pts],(255,255,255))
    #cv2.imwrite(join(MASK_DIR,_name+".png"),mask)
    cv2.imshow("image",mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def draw_and_save_filledPolygon2(_image:image):
    mask = np.zeros((_image.width, _image.height, 3), np.uint8)
    
    points_list = _image.points
    for points in points_list:
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(mask,[pts],(255,255,255))
    #cv2.imwrite(join(MASK_DIR,_name+".png"),mask)
    cv2.imshow("image",mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def draw_filledPolygon_on_image(_image:image):
    png = cv2.imread("cfc_000234.jpg")
    
    copy_png = png.copy()
    
    points_list = _image.points
    for points in points_list:
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1,1,2))
        white = (255,255,255)
        green = (153,255,51)
        purple = (204,0,255)
        cv2.fillPoly(copy_png,[pts],green)
        opacity = 0.4
        cv2.addWeighted(copy_png,opacity,png,1-opacity,0,png)
    #cv2.imwrite(join(MASK_DIR,_name+".png"),mask)
    cv2.imshow("image",png)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#draw_filledPolygon_on_image(im1) 
    
import re

pattern = r'\..*'
test_string = 'cfc_000235.png.json'
result = re.search(pattern, test_string)
result.group()
print(result.group())

name = 'cfc_000235.png.json.png'
name = re.sub(pattern, '', name)
print(name)

    
    
# draw_and_save_filledPolygon2(im1)
    
#draw_and_save_filledPolygon(200, 200, [[200,0],[200,100],[150,100],[150,0]],"image1")
#draw_and_save_filledPolygon(200, 200, [[0,0],[0,100],[50,100],[50,0]],"image1")



#for image in images:
    #draw_and_save_filledPolygon(image.width, image.height, image.points, image.name)
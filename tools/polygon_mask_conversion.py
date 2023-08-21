import argparse
import json
import os
import time
import cv2

from PIL import Image
from tqdm import tqdm
from datetime import date

import numpy as np
import matplotlib as plt

import sys
sys.path.append('.')
from anylabeling.app_info import __version__

#======================================================================= Usage ========================================================================#
#                                                                                                                                                      #
#-------------------------------------------------------------------- mask2poly  ----------------------------------------------------------------------#
# python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --mode mask2poly                                                #                             
#                                                                                                                                                      #
#-------------------------------------------------------------------- poly2mask  ----------------------------------------------------------------------#
# [option1] python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --mode poly2mask                                      # 
# [option2] python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --json_path xxx_folder --mode poly2mask               #                          
#                                                                                                                                                      #
#======================================================================= Usage ========================================================================#

VERSION = __version__
IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm']

class PolygonMaskConversion():

    def __init__(self, epsilon_factor=0.001):
        self.epsilon_factor = epsilon_factor

    def reset(self):
        self.custom_data = dict(
            version=VERSION,
            flags={},
            shapes=[],
            imagePath="",
            imageData=None,
            imageHeight=-1,
            imageWidth=-1
        )

    def get_image_size(self, image_file):
        with Image.open(image_file) as img:
            width, height = img.size
            return width, height

    def mask_to_polygon(self, img_file, mask_file, json_file):
        self.reset()
        binary_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 5:
                continue
            shape = {
                "label": "object",
                "text": "",
                "points": [],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            for point in approx:
                x, y = point[0].tolist()
                shape["points"].append([x, y])
            self.custom_data['shapes'].append(shape)

        image_width, image_height = self.get_image_size(img_file)
        self.custom_data['imagePath'] = os.path.basename(img_file)
        self.custom_data['imageHeight'] = image_height
        self.custom_data['imageWidth'] = image_width

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def polygon_to_mask(self, img_file, mask_file, json_file):

        with open(json_file, 'r') as f:
            data = json.load(f)
        polygons = []
        for shape in data['shapes']:
            points = shape['points']
            polygon = []
            for point in points:
                x, y = point
                polygon.append((x, y))
            polygons.append(polygon)

        image_width, image_height = self.get_image_size(img_file)
        image_shape = (image_height, image_width)
        binary_mask = np.zeros(image_shape, dtype=np.uint8)
        for polygon_points in polygons:
            np_polygon = np.array(polygon_points, np.int32)
            np_polygon = np_polygon.reshape((-1, 1, 2))
            cv2.fillPoly(binary_mask, [np_polygon], color=255)
        cv2.imwrite(mask_file, binary_mask)


def main():
    parser = argparse.ArgumentParser(description='Polygon Mask Conversion')

    parser.add_argument('--img_path', help='Path to image directory')
    parser.add_argument('--mask_path', help='Path to mask directory')
    parser.add_argument('--json_path', default='', help='Path to json directory')
    parser.add_argument('--epsilon_factor', default=0.001, type=float, 
                        help='Control the level of simplification when converting a polygon contour to a simplified version')
    parser.add_argument('--mode', choices=['mask2poly', 'poly2mask'], required=True,
                        help='Choose the conversion mode what you need')
    args = parser.parse_args()

    print(f"Starting conversion to {args.mode}...")
    start_time = time.time()

    converter = PolygonMaskConversion(args.epsilon_factor)

    if args.mode == "mask2poly":
        file_list = os.listdir(args.mask_path)
        for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='blue'):
            img_file = os.path.join(args.img_path, file_name)
            mask_file = os.path.join(args.mask_path, file_name)
            json_file = os.path.join(args.img_path, os.path.splitext(file_name)[0]+'.json')
            converter.mask_to_polygon(img_file, mask_file, json_file)
    elif args.mode == "poly2mask":
        os.makedirs(args.mask_path, exist_ok=True)
        file_list = os.listdir(args.img_path)
        for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='blue'):
            base_name, suffix = os.path.splitext(file_name)
            if suffix.lower() not in IMG_FORMATS:
                continue
            img_file = os.path.join(args.img_path, file_name)
            if not args.json_path:
                json_file = os.path.join(args.img_path, base_name+'.json')
            else:
                json_file = os.path.join(args.json_path, base_name+'.json')
            mask_file = os.path.join(args.mask_path, file_name)
            converter.polygon_to_mask(img_file, mask_file, json_file)


    end_time = time.time()
    print(f"Conversion completed successfully!")
    print(f"Conversion time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()

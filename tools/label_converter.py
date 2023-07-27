import argparse
import json
import os
import time

from PIL import Image
from tqdm import tqdm
from datetime import date

import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

#======================================================================= Usage ========================================================================#
#                                                                                                                                                      #
#-------------------------------------------------------------------- custom2voc  ---------------------------------------------------------------------#
# python tools/label_converter.py --src_path xxx_folder --dst_path xxx_folder --classes xxx.txt --mode custom2voc                                      #                             
#                                                                                                                                                      #
#-------------------------------------------------------------------- voc2custom  ---------------------------------------------------------------------#
# python tools/label_converter.py --src_path xxx_folder --img_path xxx_folder --classes xxx.txt --mode custom2voc                                      #
#                                                                                                                                                      #
#-------------------------------------------------------------------- custom2yolo  --------------------------------------------------------------------#
# python tools/label_converter.py --src_path xxx_folder --dst_path xxx_folder --classes xxx.txt --mode custom2yolo                                     #                             
#                                                                                                                                                      #
#-------------------------------------------------------------------- yolo2custom  --------------------------------------------------------------------#
# python tools/label_converter.py --src_path xxx_folder --img_path xxx_folder --img_path xxx_folder --classes xxx.txt --mode custom2voc                #
#                                                                                                                                                      #
#-------------------------------------------------------------------- custom2coco  --------------------------------------------------------------------#
# python tools/label_converter.py --src_path xxx_folder --dst_path xxx_folder --classes xxx.txt --mode custom2coco                                     #                             
#                                                                                                                                                      #
#-------------------------------------------------------------------- coco2custom  --------------------------------------------------------------------#
# python tools/label_converter.py --src_path xxx.json --dst_path xxx_folder --img_path xxx_folder --classes xxx.txt --mode coco2custom                 #                             
#                                                                                                                                                      #
#======================================================================= Usage ========================================================================#


VERSION = "0.2.23"

class LabelConverter:

    def __init__(self, classes_file):

        if classes_file:
            with open(classes_file, 'r') as f:
                self.classes = f.read().splitlines()
        else:
            self.classes = []

    def custom_to_voc2017(self, input_file, output_dir):
        with open(input_file, 'r') as f:
            data = json.load(f)

        image_path = data['imagePath']
        image_width = data['imageWidth']
        image_height = data['imageHeight']

        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = os.path.dirname(output_dir)
        ET.SubElement(root, 'filename').text = os.path.basename(image_path)
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(image_width)
        ET.SubElement(size, 'height').text = str(image_height)
        ET.SubElement(size, 'depth').text = '3'

        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            xmin = str(points[0][0])
            ymin = str(points[0][1])
            xmax = str(points[1][0])
            ymax = str(points[1][1])

            object_elem = ET.SubElement(root, 'object')
            ET.SubElement(object_elem, 'name').text = label
            ET.SubElement(object_elem, 'pose').text = 'Unspecified'
            ET.SubElement(object_elem, 'truncated').text = '0'
            ET.SubElement(object_elem, 'difficult').text = '0'
            bndbox = ET.SubElement(object_elem, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = xmin
            ET.SubElement(bndbox, 'ymin').text = ymin
            ET.SubElement(bndbox, 'xmax').text = xmax
            ET.SubElement(bndbox, 'ymax').text = ymax

        xml_string = ET.tostring(root, encoding='utf-8')
        dom = minidom.parseString(xml_string)
        formatted_xml = dom.toprettyxml(indent='  ')

        with open(output_dir, 'w') as f:
            f.write(formatted_xml)

    def voc2017_to_custom(self, input_file, output_file):
        self.reset()

        tree = ET.parse(input_file)
        root = tree.getroot()

        image_path = root.find('filename').text
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)

        self.custom_data['imagePath'] = image_path
        self.custom_data['imageHeight'] = image_height
        self.custom_data['imageWidth'] = image_width

        for obj in root.findall('object'):
            label = obj.find('name').text
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)

            shape = {
                "label": label,
                "text": "",
                "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }

            self.custom_data['shapes'].append(shape)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def custom_to_yolov5(self, input_file, output_file):
        with open(input_file, 'r') as f:
            data = json.load(f)

        image_width = data['imageWidth']
        image_height = data['imageHeight']

        with open(output_file, 'w') as f:
            for shape in data['shapes']:
                label = shape['label']
                points = shape['points']

                class_index = self.classes.index(label)

                x_center = (points[0][0] + points[1][0]) / (2 * image_width)
                y_center = (points[0][1] + points[1][1]) / (2 * image_height)
                width = abs(points[1][0] - points[0][0]) / image_width
                height = abs(points[1][1] - points[0][1]) / image_height

                f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

    def yolov5_to_custom(self, input_file, output_file, image_file):
        self.reset()

        with open(input_file, 'r') as f:
            lines = f.readlines()

        image_width, image_height = self.get_image_size(image_file)

        for line in lines:
            line = line.strip().split(' ')
            class_index = int(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            width = float(line[3])
            height = float(line[4])

            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)

            label = self.classes[class_index]

            shape = {
                "label": label,
                "text": None,
                "points": [[x_min, y_min], [x_max, y_max]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }

            self.custom_data['shapes'].append(shape)

        self.custom_data['imagePath'] = os.path.basename(image_file)
        self.custom_data['imageHeight'] = image_height
        self.custom_data['imageWidth'] = image_width

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def custom_to_coco(self, input_path, output_path):
        coco_data = {
            "info": {
                "year": 2023,
                "version": "1.0",
                "description": "COCO Label Conversion",
                "contributor": "CVHub",
                "url": "https://github.com/CVHub520/X-AnyLabeling",
                "date_created": str(date.today())
            },
            "licenses": [
                {
                    "id": 1,
                    "url": "https://www.gnu.org/licenses/gpl-3.0.html",
                    "name": "GNU GENERAL PUBLIC LICENSE Version 3"
                }
            ],
            "categories": [],
            "images": [],
            "annotations": []
        }

        for i, class_name in enumerate(self.classes):
            coco_data['categories'].append({
                "id": i+1,
                "name": class_name,
                "supercategory": ""
            })

        image_id = 0
        annotation_id = 0

        file_list = os.listdir(input_path)
        for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='green'):

            image_id += 1

            input_file = os.path.join(input_path, file_name)
            with open(input_file, 'r') as f:
                data = json.load(f)

            image_path = data['imagePath']
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            coco_data['images'].append({
                "id": image_id,
                "file_name": image_name,
                "width": data['imageWidth'],
                "height": data['imageHeight'],
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })

            for shape in data['shapes']:
                annotation_id += 1
                label = shape['label']
                points = shape['points']
                class_id = self.classes.index(label)
                x_min = min(points[0][0], points[1][0])
                y_min = min(points[0][1], points[1][1])
                x_max = max(points[0][0], points[1][0])
                y_max = max(points[0][1], points[1][1])
                width = x_max - x_min
                height = y_max - y_min

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id+1,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                }

                coco_data['annotations'].append(annotation)

        output_file = os.path.join(output_path, "x_anylabeling_coco.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=4, ensure_ascii=False)

    def coco_to_custom(self, input_file, output_path, image_path):

        img_dic = {}
        for file in os.listdir(image_path):
            img_dic[file] = file

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not self.classes:
            for cat in data["categories"]:
                self.classes.append(cat["name"])

        total_info, label_info = {}, {}

        # map category_id to name
        for dic_info in data["categories"]:
            label_info[dic_info["id"]] = dic_info["name"]

        # map image_id to info
        for dic_info in data["images"]:
            total_info[dic_info["id"]] = {
                "imageWidth": dic_info["width"],
                "imageHeight": dic_info["height"],
                "imagePath": img_dic[dic_info["file_name"]],
                "shapes": []
            }
        
        for dic_info in data["annotations"]:

            bbox = dic_info["bbox"]
            x_min = bbox[0]
            y_min = bbox[1]
            width = bbox[2]
            height = bbox[3]
            x_max = x_min + width
            y_max = y_min + height

            shape_info = {
                "label": self.classes[dic_info["category_id"]-1],
                "text": None,
                "points": [[x_min, y_min], [x_max, y_max]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }

            total_info[dic_info["image_id"]]["shapes"].append(shape_info)
    
        for dic_info in tqdm(total_info.values(), desc='Converting files', unit='file', colour='green'):
            self.reset()
            self.custom_data["shapes"] = dic_info["shapes"]
            self.custom_data["imagePath"] = dic_info["imagePath"]
            self.custom_data["imageHeight"] = dic_info["imageHeight"]
            self.custom_data["imageWidth"] = dic_info["imageWidth"]

            output_file = os.path.join(output_path, os.path.splitext(dic_info["imagePath"])[0]+".json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

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

def main():
    parser = argparse.ArgumentParser(description='Label Converter')
    parser.add_argument('--src_path', help='Path to input directory')
    parser.add_argument('--dst_path', help='Path to output directory')
    parser.add_argument('--img_path', help='Path to image directory')
    parser.add_argument('--classes', default=None, help='Path to classes.txt file')
    parser.add_argument('--mode', 
                        choices=['custom2voc', 'voc2custom', 'custom2yolo', 'yolo2custom', 'custom2coco', 'coco2custom'])
    args = parser.parse_args()

    converter = LabelConverter(args.classes)
    
    print(f"Starting conversion to {args.mode} format...")
    start_time = time.time()

    if args.mode == "custom2voc":
        file_list = os.listdir(args.src_path)
        for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='green'):
            os.makedirs(args.dst_path, exist_ok=False)
            src_file = os.path.join(args.src_path, file_name)
            dst_file = os.path.join(args.dst_path, os.path.splitext(file_name)[0]+'.xml')
            converter.custom_to_voc2017(src_file, dst_file)
    elif args.mode == "voc2custom":
        file_list = os.listdir(args.src_path)
        for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='green'):
            src_file = os.path.join(args.src_path, file_name)
            dst_file = os.path.join(args.img_path, os.path.splitext(file_name)[0]+'.json')
            converter.voc2017_to_custom(src_file, dst_file)
    elif args.mode == "custom2yolo":
        file_list = os.listdir(args.src_path)
        for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='green'):
            os.makedirs(args.dst_path, exist_ok=False)
            src_file = os.path.join(args.src_path, file_name)
            dst_file = os.path.join(args.dst_path, os.path.splitext(file_name)[0]+'.txt')
            converter.custom_to_yolov5(src_file, dst_file)
    elif args.mode == "yolo2custom":
        img_dic = {}
        for file in os.listdir(args.img_path):
            prefix = file.split('.')[0]
            img_dic[prefix] = file
        file_list = os.listdir(args.src_path)
        for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='green'):
            src_file = os.path.join(args.src_path, file_name)
            dst_file = os.path.join(args.img_path, os.path.splitext(file_name)[0]+'.json')
            img_file = os.path.join(args.img_path, img_dic[os.path.splitext(file_name)[0]])
            converter.yolov5_to_custom(src_file, dst_file, img_file)
    elif args.mode == "custom2coco":
        os.makedirs(args.dst_path, exist_ok=False)
        converter.custom_to_coco(args.src_path, args.dst_path)
    elif args.mode == "coco2custom":
        os.makedirs(args.dst_path, exist_ok=True)
        converter.coco_to_custom(args.src_path, args.dst_path, args.img_path)

    end_time = time.time()
    print(f"Conversion completed: {args.dst_path}")
    print(f"Conversion time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()

import argparse
import json
import os
import time

import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
from datetime import date

#======================================================================= Usage ========================================================================#
#                                                                                                                                                      #
#-------------------------------------------------------------------- custom2coco ---------------------------------------------------------------------#
# python tools/label_converter.py assets/Giraffes_at_west_midlands_safari_park.json assets/custom2coco --classes assets/classes.txt --mode custom2coco #
#------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                      #
#-------------------------------------------------------------------- custom2voc  ---------------------------------------------------------------------#
# python tools/label_converter.py assets/Giraffes_at_west_midlands_safari_park.json assets/custom2voc --mode custom2voc                                #
#------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                      #
#-------------------------------------------------------------------- custom2yolo ---------------------------------------------------------------------#
# python tools/label_converter.py assets/Giraffes_at_west_midlands_safari_park.json assets/custom2yolo --classes assets/classes.txt --mode custom2yolo #
#------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                      #
#------------------------------------------------------------------- coco2custom ----------------------------------------------------------------------#
# python tools/label_converter.py path/to/coco/*.json path/to/save/folder --classes path/to/classes.txt --mode coco2custom                             #
#------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                      #
#-------------------------------------------------------------------- voc2custom  ---------------------------------------------------------------------#
# python tools/label_converter.py path/to/voc/*.xml path/to/save/folder --classes path/to/classes.txt --mode voc2custom                                #
#------------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                                                                                                                      #
#-------------------------------------------------------------------- yolo2custom ---------------------------------------------------------------------#
# python tools/label_converter.py path/to/yolo/*.txt path/to/save/folder --classes path/to/classes.txt --image /path/to/image --mode yolo2custom       #
########################################################################################################################################################


class LabelConverter:

    def __init__(self, classes_file):
        if classes_file:
            with open(classes_file, 'r') as f:
                self.classes = f.read().splitlines()
        self.custom_data = dict(
            version="0.3.0",
            flags={},
            shapes=[],
            imagePath=None,
            imageData=None,
            imageHeight=-1,
            imageWidth=-1
        )

    def custom_to_coco(self, json_file, output_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        image_width = data['imageWidth']
        image_height = data['imageHeight']

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

        for class_name in self.classes:
            coco_data['categories'].append({
                "id": len(coco_data['categories']),
                "name": class_name,
                "supercategory": ""
            })

        image_id = 0
        annotation_id = 0

        image_path = data['imagePath']
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        coco_data['images'].append({
            "id": image_id,
            "file_name": image_name,
            "width": image_width,
            "height": image_height,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })

        for shape in data['shapes']:
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
                "category_id": class_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            }

            coco_data['annotations'].append(annotation)
            annotation_id += 1

        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=4)
    
    def custom_to_voc2017(self, json_file, output_dir):
        with open(json_file, 'r') as f:
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

    def custom_to_yolov5(self, json_file, output_file):
        with open(json_file, 'r') as f:
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

    def coco_to_custom(self, input_file, output_file):
        with open(input_file, 'r') as f:
            coco_data = json.load(f)

        image_info = coco_data['images'][0]
        image_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']

        self.custom_data['imagePath'] = image_name
        self.custom_data['imageHeight'] = image_height
        self.custom_data['imageWidth'] = image_width

        for annotation in coco_data['annotations']:
            class_id = annotation['category_id']
            bbox = annotation['bbox']
            x_min = bbox[0]
            y_min = bbox[1]
            width = bbox[2]
            height = bbox[3]
            x_max = x_min + width
            y_max = y_min + height

            shape = {
                "label": self.classes[class_id],
                "text": None,
                "points": [[x_min, y_min], [x_max, y_max]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }

            self.custom_data['shapes'].append(shape)

        with open(output_file, 'w') as f:
            json.dump(self.custom_data, f, indent=2)

    def voc2017_to_custom(self, input_file, output_file):
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
                "text": None,
                "points": [[xmin, ymin], [xmax, ymax]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }

            self.custom_data['shapes'].append(shape)

        with open(output_file, 'w') as f:
            json.dump(self.custom_data, f, indent=2)

    def yolov5_to_custom(self, input_file, output_file, image_file):
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

        with open(output_file, 'w') as f:
            json.dump(self.custom_data, f, indent=4)

    def get_image_size(self, image_file):
        with Image.open(image_file) as img:
            width, height = img.size
            return width, height

    def statistics(self, input_dir, output_dir):
        image_count = 0
        class_counts = {class_name: 0 for class_name in self.classes}

        if os.path.isdir(input_dir):
            file_list = os.listdir(input_dir)
            for file_name in tqdm(file_list, desc='Processing files', unit='file', colour='blue'):
                if file_name.endswith('.json'):
                    file_path = os.path.join(input_dir, file_name)
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    image_count += 1

                    for shape in data['shapes']:
                        label = shape['label']
                        class_counts[label] += 1

            # Save statistics
            os.makedirs(output_dir, exist_ok=True)

            # Save image count
            with open(os.path.join(output_dir, 'image_count.txt'), 'w') as f:
                for class_name, count in class_counts.items():
                    f.write(f"{class_name}: {count}\n")
                f.write(f"Total Images: {image_count}")

            print(f"Statistics saved to: {output_dir}")
        else:
            print(f"Invalid input directory: {input_dir}")


def main():
    parser = argparse.ArgumentParser(description='Label Converter')
    parser.add_argument('input', help='Path to input file or directory')
    parser.add_argument('output', help='Path to output directory')
    parser.add_argument('--image', help='Path to image file or directory')
    parser.add_argument('--classes', default=None, help='Path to classes.txt file')
    parser.add_argument('--mode', 
                        choices=['stats', 'custom2coco', 'custom2voc', 'custom2yolo', 'coco2custom', 'voc2custom', 'yolo2custom'], 
                        default='stats',
                        help='Output format (coco, voc2017, yolov5, custom)')
    args = parser.parse_args()

    converter = LabelConverter(args.classes)

    os.makedirs(args.output, exist_ok=True)
    print(f"Starting conversion to {args.mode} format...")
    start_time = time.time()
    
    if os.path.isfile(args.input):
        file_name = os.path.basename(args.input)
        base_name = os.path.splitext(file_name)[0]
        # Single file conversion
        if args.mode == 'custom2coco':
            output_dir = os.path.join(args.output, base_name+'.json')
            converter.custom_to_coco(args.input, output_dir)
        elif args.mode == 'custom2voc':
            output_dir = os.path.join(args.output, base_name+'.xml')
            converter.custom_to_voc2017(args.input, output_dir)
        elif args.mode == 'custom2yolo':
            output_dir = os.path.join(args.output, base_name+'.txt')
            converter.custom_to_yolov5(args.input, output_dir)
        elif args.mode == 'coco2custom':
            output_file = os.path.join(args.output, base_name+'.json')
            converter.coco_to_custom(args.input, output_file)
        elif args.mode == 'voc2custom':
            output_file = os.path.join(args.output, base_name+'.json')
            converter.voc2017_to_custom(args.input, output_file)
        elif args.mode == 'yolo2custom':
            output_file = os.path.join(args.output, base_name+'.json')
            converter.yolov5_to_custom(args.input, output_file, args.image)

    elif os.path.isdir(args.input):

        if args.mode == 'stats':
            converter.statistics(args.input, os.path.join(args.output, 'statistics'))
        else:
            # Batch conversion for all files in a directory
            file_list = os.listdir(args.input)
            for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='green'):
                if file_name.endswith('.json'):
                    base_name = os.path.splitext(file_name)[0]
                    input_dir = os.path.join(args.input, file_name)
                    if args.mode == 'custom2coco':
                        output_dir = os.path.join(args.output, base_name+'.json')
                        converter.custom_to_coco(input_dir, output_dir)
                    elif args.mode == 'custom2voc':
                        output_dir = os.path.join(args.output, base_name+'.xml')
                        converter.custom_to_voc2017(input_dir, output_dir)
                    elif args.mode == 'custom2yolo':
                        output_dir = os.path.join(args.output, base_name+'.txt')
                        converter.custom_to_yolov5(input_dir, output_dir)
                    elif args.mode == 'coco2custom':
                        output_file = os.path.join(args.output, base_name+'.json')
                        converter.coco_to_custom(args.input, output_file)
                    elif args.mode == 'voc2custom':
                        output_file = os.path.join(args.output, base_name+'.json')
                        converter.voc2017_to_custom(args.input, output_file)
                    elif args.mode == 'yolo2custom':
                        image_file = os.path.join(args.image, base_name+'.jpg')
                        output_file = os.path.join(args.output, base_name+'.json')
                        converter.yolov5_to_custom(args.input, output_file, image_file)
                else:
                    print(f"Skipping file: {file_name} (not a valid file)")

    else:
        print(f"Invalid input: {args.input}")

    end_time = time.time()
    print(f"Conversion completed: {args.output}")
    print(f"Conversion time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
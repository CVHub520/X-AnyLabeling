import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm
from datetime import date


class LabelConverter:
    def __init__(self, classes_file):
        with open(classes_file, 'r') as f:
            self.classes = f.read().splitlines()

    def to_coco(self, json_file, output_file):
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

    def to_voc2017(self, json_file, output_dir):
        with open(json_file, 'r') as f:
            data = json.load(f)

        image_path = data['imagePath']
        image_width = data['imageWidth']
        image_height = data['imageHeight']

        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = output_dir
        ET.SubElement(root, 'filename').text = image_path
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

        tree = ET.ElementTree(root)
        output_file = f"{output_dir}/{image_path.replace('.jpg', '.xml')}"
        tree.write(output_file)

    def to_yolov5(self, json_file, output_file):
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
    parser.add_argument('classes_file', default='classes.txt', help='Path to classes.txt file')
    parser.add_argument('--mode', choices=['stats', 'coco', 'voc2017', 'yolov5'], default='stats',
                        help='Output format (coco, voc2017, yolov5)')
    args = parser.parse_args()

    converter = LabelConverter(args.classes_file)

    os.makedirs(args.output, exist_ok=True)
    print(f"Starting conversion to {args.mode} format...")
    start_time = time.time()
    
    if os.path.isfile(args.input):
        file_name = os.path.basename(args.input)
        base_name = os.path.splitext(file_name)[0]
        # Single file conversion
        if args.mode == 'coco':
            output_dir = os.path.join(args.output, base_name+'.json')
            converter.to_coco(args.input, output_dir)
        elif args.mode == 'voc2017':
            output_dir = os.path.join(args.output, base_name+'.xml')
            converter.to_voc2017(args.input, output_dir)
        elif args.mode == 'yolov5':
            output_dir = os.path.join(args.output, base_name+'.txt')
            converter.to_yolov5(args.input, output_dir)

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
                    if args.mode == 'coco':
                        output_dir = os.path.join(args.output, base_name+'.json')
                        converter.to_coco(input_dir, output_dir)
                    elif args.mode == 'voc2017':
                        output_dir = os.path.join(args.output, base_name+'.xml')
                        converter.to_voc2017(input_dir, output_dir)
                    elif args.mode == 'yolov5':
                        output_dir = os.path.join(args.output, base_name+'.txt')
                        converter.to_yolov5(input_dir, output_dir)
                else:
                    print(f"Skipping file: {file_name} (not a JSON file)")
    else:
        print(f"Invalid input: {args.input}")

    end_time = time.time()
    print(f"Conversion completed: {args.output}")
    print(f"Conversion time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
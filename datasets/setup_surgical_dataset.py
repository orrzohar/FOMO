# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------

import xml.etree.ElementTree as ET
import os

def calculate_bounding_box(points):
    x_coordinates = [point[0] for point in points]
    y_coordinates = [point[1] for point in points]
    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)

def convert_annotation(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Create new XML structure
    annotation = ET.Element('annotation')
    filename = ET.SubElement(annotation, 'filename')
    filename.text = root.find('.//filename').text.split()[0]  # Updated to split the filename text

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = root.find('.//imagesize/ncols').text
    ET.SubElement(size, 'height').text = root.find('.//imagesize/nrows').text
    ET.SubElement(size, 'depth').text = '3'  # Assuming 3 for RGB

    for obj in root.findall('.//object'):
        object_tag = ET.SubElement(annotation, 'object')
        ET.SubElement(object_tag, 'name').text = obj.find('name').text
        ET.SubElement(object_tag, 'difficult').text = '0'

        # Calculate bounding box
        polygon = obj.find('polygon')
        points = [(int(pt.find('x').text), int(pt.find('y').text)) for pt in polygon.findall('pt')]
        xmin, ymin, xmax, ymax = calculate_bounding_box(points)

        bndbox = ET.SubElement(object_tag, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    # Write to new file
    tree = ET.ElementTree(annotation)
    tree.write(output_file, encoding='utf-8')

def convert_all_annotations(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            convert_annotation(input_file, output_file)
            print(f"Converted {filename}")

# Example usage
if __name__ == '__main__':
    output_directory = 'RWD/Annotations/Surgical'
    
    input_directory = 'ROOT/Surgical/TestSet-Annotations'
    convert_all_annotations(input_directory, output_directory)

    input_directory = 'ROOT/Surgical/TrainSet-Annotations'
    convert_all_annotations(input_directory, output_directory)



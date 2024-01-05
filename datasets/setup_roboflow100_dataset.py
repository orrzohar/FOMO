# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------

import os
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import random
import xml.etree.cElementTree as ET
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('RWD - Dataset Generation', add_help=False)
    parser.add_argument('--dataset', default='Aquatic', type=str)
    return parser


def dict_to_tuple(dictionary):
    """
    Converts a dictionary into a tuple of sorted key-value pairs.

    Args:
        dictionary (dict): Input dictionary.

    Returns:
        tuple: Sorted key-value pairs.
    """
    return tuple(sorted(dictionary.items()))


def get_unique_dicts(list_of_dicts):
    """
    Returns a list of unique dictionaries from a list of dictionaries.

    Args:
        list_of_dicts (list): List of dictionaries.

    Returns:
        list: List of unique dictionaries.
    """
    unique_dicts = list(set(dict_to_tuple(d) for d in list_of_dicts))
    return [dict(d) for d in unique_dicts]


def split_dict_categories(dict_categories):
    # Sort the dictionary by the values in descending order
    sorted_categories = sorted(dict_categories.items(), key=lambda x: x[1], reverse=True)

    # Calculate the index for splitting the categories
    split_index = len(sorted_categories) // 2

    # Get the categories with the highest num_instances
    high_categories = [category for category, _ in sorted_categories[:split_index]]

    # Get the categories with the lowest num_instances
    low_categories = [category for category, _ in sorted_categories[split_index:]]

    return high_categories, low_categories


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b


def box_xywh_to_xyxy(x):
    x, y, w, h = x
    b = [(x), (y), (x+w), (y+h)]
    return b


def coco_to_voc_detection(coco_annotation_file, target_folder):
    coco_instance = COCO(coco_annotation_file)
    total_num_imgs = 0
    for index, image_id in enumerate(coco_instance.imgToAnns):
        # import pdb;pdb.set_trace()
        image_details = coco_instance.imgs[image_id]
        annotation_el = ET.Element('annotation')
        if 'coco_url' in image_details.keys():
            ET.SubElement(annotation_el, 'filename').text = image_details['coco_url'].split('/')[-1]
        else:
            ET.SubElement(annotation_el, 'filename').text = image_details['file_name']


        size_el = ET.SubElement(annotation_el, 'size')
        ET.SubElement(size_el, 'width').text = str(image_details['width'])
        ET.SubElement(size_el, 'height').text = str(image_details['height'])
        ET.SubElement(size_el, 'depth').text = str(3)

        for annotation in coco_instance.imgToAnns[image_id]:
            object_el = ET.SubElement(annotation_el, 'object')
            # import pdb;pdb.set_trace()
            ET.SubElement(object_el,'name').text = coco_instance.cats[annotation['category_id']]['name']
            # ET.SubElement(object_el, 'name').text = 'unknown'
            ET.SubElement(object_el, 'difficult').text = '0'
            bb_el = ET.SubElement(object_el, 'bndbox')
            ET.SubElement(bb_el, 'xmin').text = str(int(annotation['bbox'][0] + 1.0))
            ET.SubElement(bb_el, 'ymin').text = str(int(annotation['bbox'][1] + 1.0))
            ET.SubElement(bb_el, 'xmax').text = str(int(annotation['bbox'][0] + annotation['bbox'][2] + 1.0))
            ET.SubElement(bb_el, 'ymax').text = str(int(annotation['bbox'][1] + annotation['bbox'][3] + 1.0))

        if 'coco_url' in image_details.keys():
            ET.SubElement(annotation_el, 'filename').text = image_details['coco_url'].split('/')[-1]
            ET.ElementTree(annotation_el).write(
                os.path.join(target_folder, image_details['coco_url'].split('/')[-1].split('.')[0] + '.xml'))

        else:
            ET.ElementTree(annotation_el).write(os.path.join(target_folder, image_details['file_name'][:-4] + '.xml'))
        total_num_imgs+=1
        if index % 10000 == 0:
            print('Processed ' + str(index) + ' images.')
    print('created', total_num_imgs, 'annotations')


# Define paths to input and output directories
def image_condition_file(ann_file, output_filename):
    # Load the COCO- annotations for the training dataset
    coco = COCO(ann_file)
    # Get the list of class IDs in the dataset
    class_ids = sorted(coco.getCatIds())
    num_instances_per_class = 100
    # Create dictionary to store output file paths
    output_filepaths = {}
    # Loop over classes and randomly select instances to crop and save
    for class_id in tqdm(class_ids, desc='[Running on classID]'):
        # Get the list of image IDs that contain objects of this class
        img_ids = coco.getImgIds(catIds=class_id)

        # Create list to store file names of cropped images for this class
        num_inst = 0
        # Loop over selected image IDs and crop and save instances of this class
        tmp = []
        for img_id in img_ids:
            if num_inst >= num_instances_per_class:
                break
            # Load the original image
            if not isinstance(img_id, list):
                img_id = [img_id]

            img_info = coco.loadImgs(img_id)[0]
            if 'file_name' in img_info.keys():
                img_name = img_info['file_name']
            else:
                img_name = img_info['coco_url'].split('/')[-1]


            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=class_id)
            anns = coco.loadAnns(ann_ids)
            anns = [a for a in anns if a['category_id']==class_id]

            if len(anns)>0:
                bboxes = np.array([a['bbox'] for a in anns])

                bboxes = bboxes/np.array([img_info['width'], img_info['height'], img_info['width'], img_info['height']])
                for bbox in bboxes:
                    tmp.append([img_name, box_xywh_to_xyxy(bbox)])
                    num_inst +=1

        tmp_nbackground = [t for t in tmp if 'background' not in t[0]]
        tmp = tmp_nbackground[:8] + [t for t in tmp if 'background' in t[0]]+tmp_nbackground[8:]
        # Add the list of cropped image filenames to the output file path dictionary
        class_name = coco.loadCats(class_id)[0]['name']
        print('found', num_inst, 'instances of', class_name)
        output_filepaths[class_name] = tmp

    # Write the output file path dictionary to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(output_filepaths, f)


def split_coco_dataset(coco_dict, imageset_dir, split_ratio=0.5):
    coco = COCO(os.path.join(imageset_dir, 'annotations.json'))
    class_ids = coco.getCatIds()
    random.shuffle(class_ids)
    train_imgs=[]
    test_imgs = []
    all_imgs = coco.getImgIds()
    random.shuffle(all_imgs)

    for _ in range(100):
        for class_id in class_ids:
            img_ids = coco.getImgIds(catIds=class_id)
            img_ids = [i for i in img_ids if i in all_imgs]
            random.shuffle(img_ids)
            if len(img_ids)>=2:
                train_tmp = img_ids[0]
                test_tmp = img_ids[1]
                all_imgs= [x for x in all_imgs if (x != train_tmp and x != test_tmp)]
                test_imgs.append(test_tmp)
                train_imgs.append(train_tmp)

            elif len(img_ids)==1:
                train_tmp = img_ids[0]
                all_imgs= [x for x in all_imgs if x != train_tmp]
                train_imgs.append(train_tmp)

    # Create a dictionary to store images by category
    for img in all_imgs:
        if random.random() > split_ratio:
            train_imgs.append(img)
        else:
            test_imgs.append(img)

    # Create separate train and test dictionaries
    train_dict = {
        'images': [c for c in coco_dict['images'] if c['id'] in train_imgs],
        'annotations':  [c for c in coco_dict['annotations'] if c['image_id'] in train_imgs],
        'categories': coco_dict['categories']
    }

    test_dict = {
        'images': [c for c in coco_dict['images'] if c['id'] in test_imgs],
        'annotations': [c for c in coco_dict['annotations'] if c['image_id'] in test_imgs],
        'categories': coco_dict['categories']
    }

    with open(os.path.join(imageset_dir,'train.json'), 'w') as outfile:
        json.dump(train_dict, outfile)

    with open(os.path.join(imageset_dir,'test.json'), 'w') as outfile:
        json.dump(test_dict, outfile)

    with open(os.path.join(imageset_dir,'test.txt'), 'w') as outfile:
        for img in test_dict['images']:
            outfile.write(img['file_name'])
            outfile.write('\n')

    with open(os.path.join(imageset_dir,'train.txt'), 'w') as outfile:
        for img in train_dict['images']:
            outfile.write(img['file_name'])
            outfile.write('\n')

    with open(os.path.join(imageset_dir,'classnames.txt'), 'w') as outfile:
        for cls in coco_dict['categories']:
            outfile.write(cls['name'])
            outfile.write('\n')

    class_ids = coco.getCatIds()
    dict_categories = {coco.loadCats(class_id)[0]['name']: len(coco.getAnnIds(catIds=class_id)) for class_id in class_ids}
    known_clsnames, _ = split_dict_categories(dict_categories)

    with open(os.path.join(imageset_dir,'known_classnames.txt'), 'w') as outfile:
        for cls in known_clsnames:
            outfile.write(cls)
            outfile.write('\n')


def combine_local_json_files(json_files, output_file):
    combined_data = {"images": [], "categories": [], "annotations": []}

    # Combine the JSON files
    for file_path in json_files:
        try:
            # Read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
            # Append the data to the combined list
            for k in combined_data.keys():
                combined_data[k]+=data[k]

            print('total number of images:', len(combined_data['images']))
        except Exception as e:
            print(f"Failed to read JSON file: {file_path}")
            print(e)

    for k in data.keys():
        if k not in combined_data.keys():
            combined_data[k] = data[k]

    combined_data['categories'] = get_unique_dicts(combined_data['categories'])
    # Write the combined data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile)
    return combined_data


def main(dataset):
    ann_files = [f'ROOT/{dataset}/train/_annotations.coco.json',
                 f'ROOT/{dataset}/valid/_annotations.coco.json',
                 f'ROOT/{dataset}/test/_annotations.coco.json']

    target_folder = f'RWD/Annotations/{dataset}'
    os.makedirs(target_folder, exist_ok=True)

    for ann_file in ann_files:
        coco_to_voc_detection(ann_file, target_folder)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RWD - Dataset Generation', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args.dataset)


## NOTE: some functions I used to create the different benchmarks:
    #imageset_dir = f'data/RWD/ImageSets/{dataset}'
    #combined_data = combine_local_json_files(ann_files, os.path.join(imageset_dir, 'annotations.json'))
    #split_coco_dataset(combined_data, imageset_dir)
    #image_condition_file(ann_files[0],
    #                     os.path.join(imageset_dir, 'few_shot_data.json'))
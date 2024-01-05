# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import cv2

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def save_image_with_bbox(image, bbox, thickness=2, save_path='output_image.png'):
    height, width, _ = image.shape
    x1 = bbox[0][0]
    y1 = bbox[0][1]
    x2 = bbox[0][2]
    y2 = bbox[0][3]

    # Convert normalized coordinates to pixel coordinates
    x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    # Draw the rectangle on the image
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
    # Save the image to the specified path
    cv2.imwrite(save_path, image)


def convert_bounding_boxes(img_shape, annotations, label):
    boxes = []
    for row in annotations:
        box = BoundingBox(x1=row[0]*img_shape[0],
                          y1=row[1]*img_shape[1],
                          x2=row[2]*img_shape[0],
                          y2=row[3]*img_shape[1],
                          label=label)
        boxes.append(box)
    return BoundingBoxesOnImage(boxes, shape=img_shape)


def revert_bounding_boxes(img_shape, bbs):
    annotations = []

    for bounding_box in bbs:
        annotation = [
            bounding_box.x1 / img_shape[0],
            bounding_box.y1 / img_shape[1],
            bounding_box.x2 / img_shape[0],
            bounding_box.y2 / img_shape[1]
        ]
        annotations.append(annotation)

    return annotations


class FewShotDataset(Dataset):
    def __init__(self, dataset,
                 image_conditioned_file,
                 known_class_names,
                 num_few_shot,
                 processor,
                 task,
                 augmentation_pipline=None):

        self.image_conditioned_file = f'data/{task}/ImageSets/{dataset}/{image_conditioned_file}'
        self.image_conditioned_dir = f'data/{task}/JPEGImages/{dataset}/'

        with open(self.image_conditioned_file) as f:
            self.few_shot_dict = json.loads(f.read())

        self.known_class_names = known_class_names
        self.num_few_shot = num_few_shot
        self.processor = processor
        self.augmentation_pipline = augmentation_pipline
        self.data_list = [(fs[0], fs[1:], i) for i, cls in enumerate(self.known_class_names)
                          for fs in self.few_shot_dict[cls][:self.num_few_shot]]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_name, bbox, label = self.data_list[idx]
        img = Image.open(os.path.join(self.image_conditioned_dir, img_name)).convert('RGB')
        img = np.asarray(img)

        if self.augmentation_pipline is not None:
            bbs = convert_bounding_boxes(img.shape, bbox, label)
            img, bbs = self.augmentation_pipline(image=img, bounding_boxes=bbs)
            bbox = revert_bounding_boxes(img.shape, bbs)

        l = torch.zeros(len(self.known_class_names))
        l[label] = 1
        img = self.processor(query_images=[img], return_tensors="pt")
        return {"image": img["query_pixel_values"], "bbox": bbox, "label": l}

    def get_no_aug(self, idx):
        img_name, bbox, label = self.data_list[idx]
        img = Image.open(os.path.join(self.image_conditioned_dir, img_name)).convert('RGB')

        img = self.processor(query_images=[img], return_tensors="pt")
        return {"image": img["query_pixel_values"], "bbox": bbox, "label": label}


often = lambda aug: iaa.Sometimes(0.75, aug)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

aug_pipeline = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        often(iaa.Affine(
            scale={"x": (0.8, 1.3), "y": (0.8, 1.3)},
            # scale images to 80-120% of their size, individually per axis
            rotate=(-10, 10),  # rotate by -45 to +45 degrees
            shear=(-25, 25),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 4),
                   [
                       # WEAK AUGMENTERS
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)),
                           # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       # add gaussian noise to images
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Add((-10, 10), per_channel=0.5),

                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),

                       # MEDIUM AUGMENTERS
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                   ],
                   random_order=True
                   )
    ])


def collate_fn(batch):
    images = []
    bboxes = []
    labels = []

    for item in batch:
        images.append(item['image'])
        bboxes.append(item['bbox'])
        labels.append(item['label'])

    images = torch.cat(images, dim=0)
    return {'image': images, 'bbox': bboxes, 'label': labels}


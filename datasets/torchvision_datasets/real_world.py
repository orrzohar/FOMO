# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection
# Orr Zohar, Jackson Wang, Serena Yeung
# ------------------------------------------------------------------------

import functools
import torch
import os
import collections
from torchvision.datasets import VisionDataset
import xml.etree.ElementTree as ET
from PIL import Image


class RWDetection(VisionDataset):
    """`RWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 args,
                 root,
                 image_set='test',
                 transforms=None, 
                 dataset='LVIS'):
        
        super(RWDetection, self).__init__(transforms)
        self.root = os.path.join(str(root), args.data_task)
        self.p_image_set = image_set
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.transforms = transforms
        imageset_root = os.path.join(self.root, "ImageSets", dataset)
        self.image_set = os.path.join(imageset_root, image_set)
        self.args = args

        with open(os.path.join(imageset_root, args.classnames_file), 'r') as file:
            ALL_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

        with open(os.path.join(imageset_root, args.prev_classnames_file), 'r') as file:
            self.PREV_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

        CUR_KNOWN_ClASSNAMES = [cls for cls in ALL_KNOWN_CLASS_NAMES if cls not in self.PREV_KNOWN_CLASS_NAMES]

        self.KNOWN_CLASS_NAMES = self.PREV_KNOWN_CLASS_NAMES+CUR_KNOWN_ClASSNAMES
        self.CLASS_NAMES = self.PREV_KNOWN_CLASS_NAMES+CUR_KNOWN_ClASSNAMES+["unknown"]

        with open(self.image_set, 'r') as file:
            file_names = file.read().splitlines()
            
        annotation_dir = os.path.join(self.root, 'Annotations', dataset)
        
        image_dir = os.path.join(self.root, 'JPEGImages', dataset)

        self.images = [os.path.join(image_dir, x) for x in file_names]
        self.annotations = [os.path.join(annotation_dir, os.path.splitext(x)[0] + ".xml") for x in file_names]
        self.imgids = [f'{i}-^-{os.path.splitext(x)[0]}' for i, x in enumerate(file_names)]
        self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))
        self.total_num_class = len(self.KNOWN_CLASS_NAMES)
        assert (len(self.images) == len(self.annotations) == len(self.imgids))
    
    
    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):
        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())

        instances = []
        for obj in target['annotation']['object']:
            cls = obj["name"]
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0

            if cls in self.KNOWN_CLASS_NAMES:
                category_id = self.KNOWN_CLASS_NAMES.index(cls)
            else:
                category_id = len(self.KNOWN_CLASS_NAMES)#-1
                
            instance = dict(
                category_id=category_id,
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                image_id=img_id
            )
            instances.append(instance)
            
        return target, instances
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Indexin
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """

        img = Image.open(self.images[index]).convert('RGB')
        target, instances = self.load_instances(self.imgids[index])
        #instances = self.label_known_class_and_unknown(instances)
        
        w, h = map(target['annotation']['size'].get, ['width', 'height'])

        target = dict(
            image_id=torch.Tensor([ord(c) for c in self.imgids[index]]),
            labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
            area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
            boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
            orig_size=torch.as_tensor([int(h), int(w)]),
            size=torch.as_tensor([int(h), int(w)]),
            iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
        )
        
        if self.transforms[-1] is not None:
            img, target = self.transforms[-1](img, target)
        #import ipdb; ipdb.set_trace()
        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


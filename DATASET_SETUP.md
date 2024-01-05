# Dataset Preparation

## Data Structure
```
FOMO/
└── data/
    ├── OWOD/
    |   ├── JPEGImages/
    |   |   ├── SOWODB/
    |   |   └── MOWODB/
    |   ├── Annotations/
    |   |   ├── SOWODB/
    |   |   └── MOWODB/
    |   └── ImageSets/
    |       ├── SOWODB/
    |       └── MOWODB/
    └── RWD/
        ├── JPEGImages/
        |   ├── Aerial/
        |   ├── Aquatic/
        |   ├── Game/
        |   ├── Medical/
        |   └── Surgical/
        ├── Annotations
        |   ├── Aerial/
        |   ├── Aquatic/
        |   ├── Game/
        |   ├── Medical/
        |   └── Surgical/
        └── ImageSets/
            ├── Aerial/
            ├── Aquatic/
            ├── Game/
            ├── Medical/
            └── Surgical/
```

## Open World Object Detection Datasets
The splits are present inside the `data/OWOD/ImageSets/MOWODB` and `data/OWOD/ImageSets/SOWODB` folders.


1. Download the COCO Images and Annotations from [coco dataset](https://cocodataset.org/#download) into the `data/` directory.
2. Unzip train2017 and val2017 folder. The current directory structure should look like:
```
FOMO/
└── data/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```
3. Move all images from `train2017/` and `val2017/` to `JPEGImages` folder.
4. Use the code `coco2voc.py` for converting json annotations to xml files.
5. Download the PASCAL VOC 2007 & 2012 Images and Annotations from [pascal dataset](http://host.robots.ox.ac.uk/pascal/VOC/) into the `data/` directory.
6. untar the trainval 2007 and 2012 and test 2007 folders.
7. Move all the images to `JPEGImages` folder and annotations to `Annotations` folder.

NOTE: I created just one folder of all the JPEG images and Annotations, for `SOWODB` and a symbolic link
for `MOWODB`.
We follow the VOC format for data loading and evaluation.

## Real World Object Detection Datasets
Download each dataset and place them in `data/ROOT`, and re-name the directory to the one used below:

```
FOMO/       
└── data/   
    ├── OWOD/
    ├── RWD/
    └── ROOT/
        ├── Aquatic/
        ├── Aerial/
        ├── Game/
        ├── Medical/
        └── Surgical/
```


1. Aquatic: https://universe.roboflow.com/roboflow-100/aquarium-qlnqy

Download in COCO format.

2. Aerial: https://gcheng-nwpu.github.io

Download the dataset from [here](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC).

3. Game: https://universe.roboflow.com/roboflow-100/team-fight-tactics

Download in COCO format.

4. Medical: https://universe.roboflow.com/roboflow-100/x-ray-rheumatology

Download in COCO format.

5. Surgical: https://medicis.univ-rennes1.fr/software#neurosurgicaltools_dataset

Download the "NeuroSurgicalToolsDataset/NeuroSurgicalToolsDataset.zip".

### Setup
**NOTE: I created a `data/data_setup.sh` bash file to setup all the dataset if all the datasets have been downloaded into the correct folders in `ROOT`.**

To convert the annotations of the roboflow100 datasets (Aquatic,Game,Medical):

```code
python datasets/roboflow100_dataset_setup.py --dataset Aquatic
```

For the Aerial and Surgical dataset:

Simply move the xml files to the `Annotations/Surgical` and png files to `JPEGImages/Surgical`.

For the Aerial dataset, use the `Horizontal Bounding Boxes` annotations. 


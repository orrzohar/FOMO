#!/bin/bash

# Create directories for JPEG images and annotations for each dataset
declare -a DATASETS=("Aquatic" "Aerial" "Game" "Medical" "Surgical")
for dataset in "${DATASETS[@]}"; do
  mkdir -p RWD/JPEGImages/$dataset
  mkdir -p RWD/Annotations/$dataset
done

# Move and rename files for Aquatic, Game, Medical datasets
declare -a PROCESSED_DATASETS=("Aquatic" "Game" "Medical")
for dataset in "${PROCESSED_DATASETS[@]}"; do
  # Move .jpg files
  mv ROOT/$dataset/valid/*.jpg RWD/JPEGImages/$dataset/
  mv ROOT/$dataset/train/*.jpg RWD/JPEGImages/$dataset/
  mv ROOT/$dataset/test/*.jpg RWD/JPEGImages/$dataset/
  python ../datasets/setup_roboflow100_dataset.py --dataset $dataset

done

# Move files for Surgical dataset
mv ROOT/Surgical/TestSet-Images/*  RWD/JPEGImages/Surgical/
mv ROOT/Surgical/TrainSet-Images/*  RWD/JPEGImages/Surgical/
python ../datasets/setup_surgical_dataset.py

# Move files for Aerial dataset
mv ROOT/Aerial/JPEGImages-test/*  RWD/JPEGImages/Aerial/
mv ROOT/Aerial/JPEGImages-trainval/*  RWD/JPEGImages/Aerial/
mv ROOT/Aerial/Annotations/'Horizontal Bounding Boxes'/*  RWD/Annotations/Aerial/


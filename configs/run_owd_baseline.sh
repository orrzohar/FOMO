#!/bin/bash

model=base-patch16
NUM_FS=100
TCP_PORT=29502
DATASET=MOWODB

declare -A BATCH_SIZEs
BATCH_SIZEs["base-patch16"]=10
BATCH_SIZEs["large-patch14"]=5

declare -A IMAGE_SIZEs
IMAGE_SIZEs["base-patch16"]=768
IMAGE_SIZEs["large-patch14"]=840

BATCH_SIZE=${BATCH_SIZEs[$model]}
IMAGE_SIZE=${IMAGE_SIZEs[$model]}

output_file_name="owod_${model}.csv"
test_set="test"
train_set="train"
unk_filenames="_classnames_groundtruth"

python main.py --model_name "google/owlvit-${model}" --batch_size $BATCH_SIZE \
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --TCP $TCP_PORT \
  --dataset $DATASET --image_resize $IMAGE_SIZE --output_file output_file_name \
  --classnames_file 't1_known.txt' --prev_classnames_file 't1_known.txt'\
  --test_set "${test_set}.txt" --train_set "${train_set}.txt" --unk_methods 'None' --unk_method 'None' \
  --unknown_classnames_file "t1_unknown$unk_filenames.txt" --unk_proposal \
  --output_file "owod_${model}_t1.csv" --data_task 'OWOD'


python main.py --model_name "google/owlvit-${model}" --batch_size $BATCH_SIZE \
  --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 --TCP $TCP_PORT \
  --dataset $DATASET --image_resize $IMAGE_SIZE --output_file output_file_name \
  --classnames_file 't2_known.txt' --prev_classnames_file 't1_known.txt' \
  --test_set "${test_set}.txt" --train_set "${train_set}.txt" --unk_methods 'None' --unk_method 'None' \
  --unknown_classnames_file "t2_unknown$unk_filenames.txt" --unk_proposal \
  --output_file "owod_${model}_t2.csv" --data_task 'OWOD'


python main.py --model_name "google/owlvit-${model}" --batch_size $BATCH_SIZE \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --TCP $TCP_PORT \
  --dataset $DATASET --image_resize $IMAGE_SIZE --output_file output_file_name \
  --classnames_file 't3_known.txt' --prev_classnames_file 't2_known.txt' \
  --test_set "${test_set}.txt" --train_set "${train_set}.txt" --unk_methods 'None' --unk_method 'None' \
  --unknown_classnames_file "t3_unknown$unk_filenames.txt" --unk_proposal \
  --output_file "owod_${model}_t3.csv" --data_task 'OWOD'


python main.py --model_name "google/owlvit-${model}" --batch_size $BATCH_SIZE \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --TCP $TCP_PORT \
  --dataset $DATASET --image_resize $IMAGE_SIZE --output_file output_file_name \
  --classnames_file 't4_known.txt' --prev_classnames_file 't3_known.txt' \
  --test_set "${test_set}.txt" --train_set "${train_set}.txt" --unk_methods 'None' --unk_method 'None' \
  --output_file "owod_${model}_t4.csv" --data_task 'OWOD'

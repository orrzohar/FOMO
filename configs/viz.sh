#!/bin/bash


declare -a DATASETS=("Aquatic" "Surgical" "Medical" "Aerial" "Game")

declare -A CUR_INTRODUCED_CLS
CUR_INTRODUCED_CLS["Aerial"]=10
CUR_INTRODUCED_CLS["Surgical"]=6
CUR_INTRODUCED_CLS["Medical"]=6
CUR_INTRODUCED_CLS["Aquatic"]=4
CUR_INTRODUCED_CLS["Game"]=30

model="google/owlvit-large-patch14"
num_shots=100
BATCH_SIZE=5
IMAGE_SIZE=840

for dataset in "${DATASETS[@]}"; do
    cur_cls=${CUR_INTRODUCED_CLS[$dataset]}

    python main.py --model_name $model --num_few_shot $num_shots --batch_size $BATCH_SIZE \
          --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS $cur_cls --TCP 28432 --dataset $dataset \
          --image_conditioned --image_resize $IMAGE_SIZE --use_attributes \
          --att_refinement --att_adapt --att_selection --viz --output_dir "viz/$dataset"
  done

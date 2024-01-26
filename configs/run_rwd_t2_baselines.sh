#!/bin/bash

set -e
cur_fname="$(basename $0 .sh)"
script_name=$(basename $0)

# Cluster parameters
partition=""
account=""


# Initialize TCP port and counter
TCP_INIT=29060
counter=0
current_date=$(date +'%Y-%m-%d_%H-%M-%S')
save_dict="tmp/imagenet/t_2"


# Declare arrays for different configurations
declare -a DATASETS=("Aerial" "Surgical" "Medical" "Medical" "Game")
declare -a MODELS=("google/owlvit-base-patch16" "google/owlvit-large-patch14")
declare  -a UNK_CLASS_FILES=("None" "unknown_classnames_ground_truth.txt" "unknown_classnames_imagenet.txt" "unknown_classnames.txt")


# Declare associative array for CUR_INTRODUCED_CLS per dataset
declare -A PREV_INTRODUCED_CLS
PREV_INTRODUCED_CLS["Aerial"]=10
PREV_INTRODUCED_CLS["Surgical"]=6
PREV_INTRODUCED_CLS["Medical"]=6
PREV_INTRODUCED_CLS["Aquatic"]=4
PREV_INTRODUCED_CLS["Game"]=30


declare -A CUR_INTRODUCED_CLS
CUR_INTRODUCED_CLS["Aerial"]=10
CUR_INTRODUCED_CLS["Surgical"]=7
CUR_INTRODUCED_CLS["Medical"]=6
CUR_INTRODUCED_CLS["Aquatic"]=3
CUR_INTRODUCED_CLS["Game"]=29

declare -A BATCH_SIZEs
BATCH_SIZEs["google/owlvit-base-patch16"]=4
BATCH_SIZEs["google/owlvit-large-patch14"]=2

declare -A IMAGE_SIZEs
IMAGE_SIZEs["google/owlvit-base-patch16"]=768
IMAGE_SIZEs["google/owlvit-large-patch14"]=840

# Loop through each configuration
for unk_class_file in "${UNK_CLASS_FILES[@]}"; do
for dataset in "${DATASETS[@]}"; do
  cur_cls=${CUR_INTRODUCED_CLS[$dataset]}
  prev_cls=${PREV_INTRODUCED_CLS[$dataset]}

  for model in "${MODELS[@]}"; do
    BATCH_SIZE=${BATCH_SIZEs[$model]}
    IMAGE_SIZE=${IMAGE_SIZEs[$model]}
    tcp=$((TCP_INIT + counter))

    # Construct the command to run
    cmd="python main.py --model_name \"$model\" --batch_size $BATCH_SIZE \
    --PREV_INTRODUCED_CLS $prev_cls --CUR_INTRODUCED_CLS $cur_cls --TCP $tcp --dataset $dataset \
    --image_resize $IMAGE_SIZE --classnames_file 'classnames.txt' \
    --unknown_classnames_file $unk_class_file --unk_proposal  \
    --unk_methods 'None' --unk_method 'None' \
    --output_dir $save_dict  --output_file $model-$dataset-$unk_class_file.csv"

    echo "Constructed Command: $cmd"

    # Uncomment below to submit the job
    sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=${dataset}-${cur_fname}-${unk_class_file}
#SBATCH --output=slurm_logs/${dataset}-${current_date}-${cur_fname}-${unk_class_file}-%j-out.txt
#SBATCH --error=slurm_logs/${dataset}-${current_date}-${cur_fname}-${unk_class_file}-%j-err.txt
#SBATCH --mem=48gb
#SBATCH -c 2
#SBATCH --gres=gpu:a6000
#SBATCH -p $partition
#SBATCH -A $account
#SBATCH --time=48:00:00
#SBATCH --ntasks=1 
echo \"$cmd\"
# Uncomment below to actually run the command
eval \"$cmd\"
"

    counter=$((counter + 1))
  done
done
done

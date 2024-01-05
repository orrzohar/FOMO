#!/bin/bash

declare -a DATASETS=("Aerial" "Surgical" "Medical" "Aquatic" "Game")

for dataset in "${DATASETS[@]}"; do
  rm -r $dataset
  mkdir $dataset
  done
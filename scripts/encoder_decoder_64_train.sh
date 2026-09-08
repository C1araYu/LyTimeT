#!/bin/bash

dataset=$1
gpu=$2

echo "===== Training encoder-decoder-64 model on: $dataset (gpu id: $gpu) ====="

screen -S train-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model64/config1.yaml NA > ../logs/"$dataset"_train_config1.log 2>&1; \
                                           CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model64/config2.yaml NA > ../logs/"$dataset"_train_config2.log 2>&1; \
                                           CUDA_VISIBLE_DEVICES="$gpu" python ../main.py ../configs/"$dataset"/model64/config3.yaml NA > ../logs/"$dataset"_train_config3.log 2>&1; \
                                           exec sh";
#!/bin/bash

dataset=$1
gpu=$2

echo "===== Gathering encoder-decoder-64 model on: $dataset (gpu id: $gpu) ====="

# Start gathering processes in a screen session and log outputs
screen -S gather-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config1.yaml ../logs/"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints NA eval-encoder-decoder-train > ../logs/"$dataset"_gather_config1.log 2>&1; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config2.yaml ../logs/"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints NA eval-encoder-decoder-train > ../logs/"$dataset"_gather_config2.log 2>&1; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/model64/config3.yaml ../logs/"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints NA eval-encoder-decoder-train > ../logs/"$dataset"_gather_config3.log 2>&1; \
                                            exec sh";

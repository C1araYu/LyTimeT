#!/bin/bash

dataset=$1
gpu=$2

echo "===== Gathering refine-64 model on: $dataset (gpu id: $gpu) ====="

# Log outputs to ../logs/<dataset>_refine_gather_configX.log
screen -S gather-"$dataset"-64 -dm bash -c "CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config1.yaml ../logs/"$dataset"_encoder-decoder-64_1/lightning_logs/checkpoints ../logs/"$dataset"_refine-64_1/lightning_logs/checkpoints eval-refine-train > ../logs/"$dataset"_refine_gather_config1.log 2>&1; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config2.yaml ../logs/"$dataset"_encoder-decoder-64_2/lightning_logs/checkpoints ../logs/"$dataset"_refine-64_2/lightning_logs/checkpoints eval-refine-train > ../logs/"$dataset"_refine_gather_config2.log 2>&1; \
                                            CUDA_VISIBLE_DEVICES="$gpu" python ../eval.py ../configs/"$dataset"/refine64/config3.yaml ../logs/"$dataset"_encoder-decoder-64_3/lightning_logs/checkpoints ../logs/"$dataset"_refine-64_3/lightning_logs/checkpoints eval-refine-train > ../logs/"$dataset"_refine_gather_config3.log 2>&1; \
                                            exec sh";

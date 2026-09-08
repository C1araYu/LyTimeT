#!/bin/bash

dataset=$1

echo "===== Evaluating metrics on: $dataset ====="

# Log outputs to ../logs/<dataset>_metrics_<task>.log
screen -S eval-"$dataset"-metrics -dm bash -c "python ../utils/smoothness.py "$dataset" ../configs/"$dataset"/refine64 > ../logs/"$dataset"_metrics_smoothness.log 2>&1; \
                                               python ../utils/mutual_info.py "$dataset" ../configs/"$dataset"/refine64 > ../logs/"$dataset"_metrics_mutual_info.log 2>&1; \
                                               exec sh";

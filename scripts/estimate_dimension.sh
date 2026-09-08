#!/bin/bash

dataset=$1
gpu=$2

echo "===== Estimating intrinsic dimension on: $dataset ====="

# Log output to ../logs/dimension_<dataset>.log
screen -S eval-"$dataset"-dimension -dm bash -c "python ../utils/dimension.py "$dataset" ../configs/"$dataset"/model64 > ../logs/dimension_$dataset.log 2>&1; \
                                                 exec sh";

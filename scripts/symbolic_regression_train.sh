#!/bin/bash

dataset=$1

echo "===== Training symbolic-regression model on: $dataset ====="

# Log outputs to ../logs/<dataset>_symbolic_regression_train_configX.log
screen -S train-"$dataset"-sr -dm bash -c "python ../regression.py ../configs/"$dataset"/regression/config1.yaml > ../logs/"$dataset"_symbolic_regression_train_config1.log 2>&1; \
                                           python ../regression.py ../configs/"$dataset"/regression/config2.yaml > ../logs/"$dataset"_symbolic_regression_train_config2.log 2>&1; \
                                           python ../regression.py ../configs/"$dataset"/regression/config3.yaml > ../logs/"$dataset"_symbolic_regression_train_config3.log 2>&1; \
                                           exec sh";

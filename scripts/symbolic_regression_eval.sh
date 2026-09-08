#!/bin/bash

dataset=$1

echo "===== Evaluating symbolic-regression model on: $dataset ====="

# Log outputs to ../logs/<dataset>_symbolic_regression_eval_configX.log
screen -S eval-"$dataset"-sr -dm bash -c "python ../eval.py ../configs/"$dataset"/regression/config1.yaml ../logs/"$dataset"_refine-64_1/symbolic_regression/model.pkl NA eval-symbolic-regression-train > ../logs/"$dataset"_symbolic_regression_eval_config1.log 2>&1; \
                                          python ../eval.py ../configs/"$dataset"/regression/config2.yaml ../logs/"$dataset"_refine-64_2/symbolic_regression/model.pkl NA eval-symbolic-regression-train > ../logs/"$dataset"_symbolic_regression_eval_config2.log 2>&1; \
                                          python ../eval.py ../configs/"$dataset"/regression/config3.yaml ../logs/"$dataset"_refine-64_3/symbolic_regression/model.pkl NA eval-symbolic-regression-train > ../logs/"$dataset"_symbolic_regression_eval_config3.log 2>&1; \
                                          exec sh";

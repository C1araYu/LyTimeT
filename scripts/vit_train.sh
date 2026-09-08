#!/bin/bash

# dataset=$1
# gpu=$2

# echo "===== Training ViT-TIDE model on: $dataset (gpu id: $gpu) ====="

# screen -S train-"$dataset"-vit -dm bash -c "CUDA_VISIBLE_DEVICES=$gpu python ../main.py ../configs/$dataset/vit/config1.yaml NA > ../logs/${dataset}_vit_train1_${int(time.time())}.log 2>&1;\
#                                             CUDA_VISIBLE_DEVICES=$gpu python ../main.py ../configs/$dataset/vit/config2.yaml NA > ../logs/${dataset}_vit_train2_${int(time.time())}.log 2>&1;\
#                                             CUDA_VISIBLE_DEVICES=$gpu python ../main.py ../configs/$dataset/vit/config3.yaml NA > ../logs/${dataset}_vit_train3_${int(time.time())}.log 2>&1; 
#                                              exec sh;"

#                                              #!/bin/bash

dataset=$1
gpu=$2

echo "===== Training ViT-TIDE model on: $dataset (gpu id: $gpu) ====="

LOG_TIME=$(date +%s)

screen -S train-"$dataset"-vit -dm bash -c "CUDA_VISIBLE_DEVICES=$gpu python ../main.py ../configs/$dataset/vit/config1.yaml NA > ../logs/${dataset}_vit_train1.log 2>&1; \
                                           CUDA_VISIBLE_DEVICES=$gpu python ../main.py ../configs/$dataset/vit/config2.yaml NA > ../logs/${dataset}_vit_train2.log 2>&1; \
                                           CUDA_VISIBLE_DEVICES=$gpu python ../main.py ../configs/$dataset/vit/config3.yaml NA > ../logs/${dataset}_vit_train3.log 2>&1; \
                                           exec sh"


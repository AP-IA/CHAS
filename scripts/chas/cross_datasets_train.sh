#!/bin/bash

# custom config
DATA=/path/to/dataset/folder
TRAINER=CHAS
DATASET=imagenet
CFG=vit_b16_cross_datasets

SHOTS=16
SEED=1
DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TASK CD
    fi
    
python3 parse_test_res.py output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/
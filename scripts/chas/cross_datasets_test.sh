#!/bin/bash

# custom config
DATA=/path/to/dataset/folder
TRAINER=CHAS

DATASET=$1

CFG=vit_b16_cross_datasets

SHOTS=16
SEED=1
MODEL_DIR=output/base2new/train_base/imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    DIR=output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
        --model-dir ${MODEL_DIR} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        TASK CD
    fi

python3 parse_test_res.py output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/
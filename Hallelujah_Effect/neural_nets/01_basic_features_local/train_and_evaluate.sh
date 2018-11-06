#!/bin/bash

#PBS -l nodes=1,gpus=2
#PBS -l walltime=00:10:00
#PBS -q v100_dev_q
#PBS -A DeepLearner
#PBS -W group_list=cascades
#PBS -M brennon@vt.edu
#PBS -m bea

cd $PBS_O_WORKDIR

# Save results to directory based on run time
export START=$(date +"%Y%m%d%H%M%S")
export MODEL_DIR=./model_trained_${START}

module purge

module load Anaconda/5.1.0
module load cuda/9.0.176
module load cudnn/7.1

source activate eim-analysis

cp -R ~/data $TMPFS

python \
    -m trainer.trainer.task \
    --train_data_paths="${TMPFS}/data/train*" \
    --eval_data_paths="${TMPFS}/data/eval*" \
    --output_dir=$MODEL_DIR \
    --train_steps=10 \
    --optimize=true

exit;

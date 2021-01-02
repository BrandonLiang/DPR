#!/bin/bash -e

# note: to use 'greadlink', you need to have coreutils installed on your MacOS ("brew install coreutils")
#       if you are on Linux, use 'readlink' instead

#SCRIPT=`greadlink -f "$0"`
SCRIPT=`readlink -f "$0"`
SCRIPT_DIR=`dirname "$SCRIPT"`
APP_HOME="$SCRIPT_DIR"

CONF=$APP_HOME/conf
PYTHON_DIR=$APP_HOME/python
ENV="$CONF"/env.sh # configuration file
source "$ENV"

DATA_DIR=$APP_HOME/data

CHECKPOINT_DIR=$APP_HOME/checkpoint
mkdir -p $CHECKPOINT_DIR/nq

STAGE=1

python -m torch.distributed.launch \
  --nproc_per_node=4 train_dense_encoder.py \
  --max_grad_norm 2.0 \
  --encoder_model_type hf_bert \
  --pretrained_model_cfg bert-base-uncased \
  --seed 12345 \
  --sequence_length 256 \
  --warmup_steps 1237 \
  --batch_size 16 \
  --do_lower_case \
  --train_file ${DATA_DIR}/retriever/nq-train.json \
  --dev_file ${DATA_DIR}/retriever/nq-dev.json \
  --output_dir $CHECKPOINT_DIR/nq \
  --learning_rate 2e-5 \
  --num_train_epochs 40 \
  --dev_batch_size 16 \
  --val_av_rank_start_epoch 30

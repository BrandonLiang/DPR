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

DATA_DIR=$APP_HOME/data/vcr
mkdir -p $DATA_DIR
CHECKPOINT_DIR=$APP_HOME/checkpoint
mkdir -p $CHECKPOINT_DIR/vcr

CHECKPOINT_MODEL=$CHECKPOINT_DIR/vcr/dpr_biencoder.33.2900

STAGE=6

if [ $STAGE -le 1 ]; then
  python $PYTHON_DIR/annotation_to_DPR_data.py \
    --train_input $TSV_DIR/train_annots.bsv \
    --val_input $TSV_DIR/val_annots.bsv \
    --tsv_output $TSV_DIR/vcr_title_text.tsv \
    --dpr_input $DATA_DIR/train.json
fi

if [ $STAGE -le 2 ]; then
  python -m torch.distributed.launch \
    --nproc_per_node=4 train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-uncased \
    --seed 12345 \
    --sequence_length 64 \
    --warmup_steps 1237 \
    --batch_size 32 \
    --do_lower_case \
    --train_file ${DATA_DIR}/train.json \
    --dev_file ${DATA_DIR}/train.json \
    --output_dir $CHECKPOINT_DIR/vcr \
    --learning_rate 2e-5 \
    --num_train_epochs 40 \
    --dev_batch_size $BATCH_SIZE \
    --val_av_rank_start_epoch 30 \
    --global_loss_buf_sz 1500000
fi

if [ $STAGE -le 3 ]; then
  python generate_dense_embeddings.py \
    --model_file $CHECKPOINT_MODEL \
    --ctx_file $TSV_DIR/vcr_title_text.tsv \
    --out_file $DATA_DIR/generated_embeddings
fi

if [ $STAGE -le 4 ]; then
  python $PYTHON_DIR/annotation_to_event_answer.py \
    --input $DATA_DIR/train.json \
    --output $DATA_DIR/vcr_event_answer.csv
fi

if [ $STAGE -le 5 ]; then
  python dense_retriever.py \
    --model_file $CHECKPOINT_MODEL \
    --ctx_file $TSV_DIR/vcr_title_text.tsv \
    --qa_file $DATA_DIR/vcr_event_answer.csv \
    --encoded_ctx_file $DATA_DIR/generated_embeddings_0.pkl \
    --out_file $DATA_DIR/retrieval_output.json \
    --n-docs 100 \
    --validation_workers 32 \
    --batch_size 64
fi

if [ $STAGE -le 6 ]; then
  python $PYTHON_DIR/DPR_retrieval_output_to_visual_comet_input.py \
    --input $DATA_DIR/retrieval_output.json \
    --prev_train $RETRIEVAL_DIR/embedding_knn_prediction_train.json \
    --prev_val $RETRIEVAL_DIR/embedding_knn_prediction_val.json \
    --train_output $RETRIEVAL_DIR/dpr_prediction_train.json \
    --val_output $RETRIEVAL_DIR/dpr_prediction_val.json
fi

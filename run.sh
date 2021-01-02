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

STAGE=1

if [ $STAGE -le 1 ]; then
  python $PYTHON_DIR/annotation_to_DPR_data.py \
    --train_input $TSV_DIR/train_annots.bsv \
    --val_input $TSV_DIR/val_annots.bsv \
    --tsv_output $TSV_DIR/vcr_title_text.tsv \
    --dpr_input $DATA_DIR/train.json
fi

#if [ $STAGE -le 2 ]; then
#fi

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

CHECKPOINT_MODEL=$CHECKPOINT_DIR/vcr/dpr_biencoder.36.2583

python get_query_encoder.py \
  --model_file $CHECKPOINT_MODEL

#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3

if [[ $# -lt 6 ]]
then
    echo "usage: $0 <KERAS_BACKEND> <theano|tensorflow> <CUDA_VISIBLE_DEVICES> (0,1,...) [main.py params]"
    python main.py -h
    exit 1
fi

export THEANO_FLAGS="device=cuda,floatX=float32,dnn.conv.algo_bwd_filter='deterministic',dnn.conv.algo_bwd_data='deterministic'"
export LIBRARY_PATH=/usr/local/cuda/lib64

export KERAS_BACKEND=$1
export CUDA_VISIBLE_DEVICES=$2

python main.py "${@:3}"

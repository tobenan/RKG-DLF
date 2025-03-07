#!/usr/bin/env bash

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-29501}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}

#PYTHONPATH="/media/ders/GDH/mmaction-master/tools":"/media/soft/gdh/anaconda3/envs/mmaction/lib/python3.7/site-packages" \
python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS --master_port=$PORT /media/ders/GDH/mmaction-master/tools/train.py $CONFIG --launcher pytorch ${@:3}
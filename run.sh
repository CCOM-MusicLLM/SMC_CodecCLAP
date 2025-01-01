#!/bin/bash
set -ex
export NCCL_SOCKET_TIMEOUT=-1
export NCCL_COMM_TIMEOUT=-1
export NCCL_DEBUG=warn
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_IB_HCA=mlx5_0

# export NCCL_ALGO=^Ring

if [ "${MULTI_NODE:-0}" = "1" ]; then
    export MY_NODE_RANK=${RANK}
    export TOTAL_NODES=${WORLD_SIZE}
    MASTER_ADDR=${MASTER_ADDR}
    MASTER_PORT=${MASTER_PORT}
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
else
    export MY_NODE_RANK=0
    export TOTAL_NODES=1
    MASTER_ADDR=localhost
    MASTER_PORT=6001
    NNODES=1
    NODE_RANK=0
fi


export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE"


cat /2214/wandb.netrc > /root/.netrc
cd /2214/dongyuanliang
export TORCH_HOME='/2214/torch'
export HF_HOME='/2214/huggingface'


/2214/conda_envs/codecclap_final/bin/torchrun $DISTRIBUTED_ARGS SMC_CodecCLAP/retrieval/smc.py -c SMC_CodecCLAP/retrieval/settings/baseline.yaml \
> SMC_CodecCLAP/train_stdout_log_${NODE_RANK}.txt \
2> SMC_CodecCLAP/train_stderr_log_${NODE_RANK}.txt
# /2214/conda_envs/musicgen/bin/python trainer.py 2>&1 | tee -a train_log_myenc32_flatten2_1min_medium_multinode_$NODE_RANK.txt
tail -f /dev/null


#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# /2214/conda_envs/musicgen/bin/torchrun $DISTRIBUTED_ARGS /2214/dongyuanliang/Megatron-LM/pretrain_gpt.py ${options}

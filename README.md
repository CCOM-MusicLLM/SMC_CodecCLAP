# SMC_CodecCLAP

This project explores the integration of Neural Audio Codecs (NACs) into Contrastive Language-Audio Pretraining (CLAP) models, demonstrating their superior feature discrimination and retrieval efficacy, and setting new benchmarks for audio representation in AI systems.

## Setup Guide
We highly recommend users to run this project under `conda` environment.

#### Prepare the environment:
To create a new environment with the necessary dependencies, run the following command:
```
conda env create --name envname --file=env.yaml
``` 

## DEMO Usage examples

### DDP on Multi-GPU nodes
To run the project on multiple GPU nodes using Distributed Data Parallel (DDP), use the following command:
```bash
torchrun --nproc_per_node=4 SMC_CodecCLAP/retrieval/smc.py -c SMC_CodecCLAP/retrieval/settings/mel.yaml
```

### CPU nodes
If you are running on CPU nodes, use the following command:
```bash
python3 SMC_CodecCLAP/retrieval/smc.py -c SMC_CodecCLAP/retrieval/settings/encodec.yaml
```
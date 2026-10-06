# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**STEP** is a unified benchmarking platform for Spiking Transformer models (NeurIPS 2025). It supports three computer vision tasks — classification, segmentation, and object detection — using Spiking Neural Networks (SNNs) built on the BrainCog framework.

## Installation

```bash
# Core SNN framework (required first)
pip install git+https://github.com/braincog-X/Brain-Cog.git

# Classification module
pip install -r cls/requirements.txt

# Detection module (Python 3.10, CUDA 11.x strongly recommended)
cd det && pip install -e .
pip install -r det/requirements/runtime.txt

# Segmentation module
cd seg && pip install -e .
pip install -r seg/requirements/runtime.txt
```

Key pinned versions: `torch==2.4.1`, `timm==0.5.4`, `spikingjelly==0.0.0.0.14`.

## Running Training

**Classification:**
```bash
cd cls
python train.py --config configs/spikformer/cifar10.yml         # static images
python train_dvs.py --config configs/spikformer/cifar10dvs.yml  # neuromorphic/DVS
python train_pc.py --config configs/spt/modelnet40.yml          # 3D point clouds
```

**Detection (multi-GPU DDP):**
```bash
cd det/tools
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./dist_train.sh ../configs/spikformer/mask_rcnn.py 8
```

**Segmentation:**
```bash
cd seg/tools
CUDA_VISIBLE_DEVICES=0 ./dist_train.sh ../configs/spikformer/config.py 1
```

**Evaluation:**
```bash
cd det/tools && python test.py ../configs/<model>/config.py <checkpoint.pth>
cd seg/tools && python test.py ../configs/<model>/config.py <checkpoint.pth>
```

## Running Tests

```bash
# Detection module has pytest config (xdoctest)
cd det && pytest tests/

# Code quality checks (configured in det/setup.cfg and seg/setup.cfg)
cd det && pre-commit run --all-files  # flake8, isort, yapf, codespell
```

## Architecture

### Module Layout

- **`cls/`** — Classification: self-contained with its own training loops, models, data loaders
- **`det/`** — Detection: fork of MMDetection with SNN backbones plugged in
- **`seg/`** — Segmentation: fork of MMSegmentation with SNN backbones plugged in

### Classification Data Flow

1. YAML config → dataset + model + hyperparameter selection
2. `timm.create_model()` selects the registered SNN model
3. `cls/data/loader.py` builds a fast-collate data loader
4. Training loop unfolds T timesteps per forward pass (T controlled by `step:` in config)
5. Spiking tensors reshape via `rearrange2node()` (T,B,C,H,W → TB,C,H,W) before neuron ops, then `rearrange2op()` to restore

### Neuron Implementations (`cls/models/utils/node.py`)

All neurons extend `BaseNode_Torch` (BrainCog). Available types (set via `node_type:` in config):
- `LIFNode`, `PLIFNode` — standard/parametric Leaky Integrate-and-Fire
- `ILIFNode`, `NILIFNode` — integer/normalized integer LIF
- `GLIFNode`, `KLIFNode`, `CLIFNode` — gated/k-based/complementary LIF
- `PSNNode`, `HHNode`, `IzhNode` — parallel spiking, Hodgkin-Huxley, Izhikevich

Surrogate gradients for backprop through spikes: `SigmoidGrad`, `AtanGrad`, etc. (`cls/models/utils/surrogate.py`).

### Spiking Transformer Block

- **SPS (Spiking Patch Splitter):** 4 stacked Conv→BN→LIF layers with max pooling for tokenization
- **SSA (Spiking Self-Attention):** standard multi-head attention with spiking activations replacing softmax

### Det/Seg Integration

SNN models are registered as MMDetection/MMSegmentation backbones via `@MODELS.register_module()`. Hierarchical configs in `_base_/` define datasets and schedules; task configs import and extend them.

### Configuration System

YAML configs (cls) and Python configs (det/seg) are the single source of truth for experiments. Key classification config fields:
- `model:` — registered model name (e.g., `spikformer_cifar`)
- `step:` — number of SNN timesteps (T)
- `node_type:` / `tau:` / `threshold:` / `act_function:` — neuron parameters
- `data_dir:` — **must be updated** to local dataset path

### Supported Models

18+ models including Spikformer (ICLR 2023), QKFormer (NeurIPS 2024), SDT, SGLFormer, SpikingResformer, Spiking Wavelet Transformer, TIM, and Spiking Point Transformer. Each has dedicated files under `cls/models/static/`, `cls/models/dvs/`, or `cls/models/pc/`, and paired YAML configs.

## Key Constraints

- Dataset paths in configs are absolute — update `data_dir:` for your environment before running
- CUDA 11.x is required for det/seg; CUDA 12.x has known compatibility issues with mmcv
- BrainCog must be installed from source (git) before any other dependency
- `timm` is pinned to `0.5.4`; the API differs significantly from newer versions

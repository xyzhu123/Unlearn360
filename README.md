# Unlearn360
This repository contains implementations for [machine unlearning](https://arxiv.org/abs/2402.08787) methods on [LLM360](https://github.com/LLM360) Models. Machine unlearning is a pre-deployment safety measure designed to remove hazardous knowledge from language models. Unlearned models are inherently safe, as they lack the knowledge to be misused. 

## Table of Contents 
- [Overview](#overview)
  - [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Training and Evaluation](#training-and-evaluation)

## Overview

### Directory Structure

``unlearn.py`` is the main entrypoint for running unlearning methods. It uses python modules in ``methods/`` and ``utils/`` folders.

The ``methods/`` folder contains the implementations for unlearning methods:
- ``training.py``: All training loop implementations
- ``utils.py``: Loss functions and other method-related utils

The ``utils/`` folder contains helper functions for model/dataset IO:
- ``dataloaders.py``: Dataloader for text datasets
- ``model_utils.py``: Model IO utils

By default, unlearned models are saved to ``models/`` folder. Please store all training datasets to the ``data/`` folder. 

> [!NOTE]
> This project uses the [bio-forget-corpus](https://huggingface.co/datasets/cais/wmdp-corpora) from the [WMDP Benchmark](https://www.wmdp.ai/) for unlearning training. Access to this dataset requires a separate request. Please follow the instructions provided [here](https://huggingface.co/datasets/cais/wmdp-corpora) to obtain the necessary permissions. By default, the dataloader is configured to load the dataset from ``data/bio_forget.jsonl``.

## Quick Start
### Setup
1. Clone and enter the repo:
    ```bash
    git clone https://github.com/xyzhu123/Unlearn360.git
    cd Unlearn360
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. To install ``lm-eval``, run the following commands visit the [official repo](https://github.com/EleutherAI/lm-evaluation-harness): 
    ```bash
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .
    ```
### Training and Evaluation
An example usage is provided in the ``max_entropy_exp.ipynb``, which can be executed with a single ``A100 80G`` GPU.
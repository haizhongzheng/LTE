# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.distributed as dist

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import model_config as MODEL_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from accelerate.utils import is_xpu_available

from lte_model_generator import model_generator_llama

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, model_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    for k, v in kwargs.items():
            setattr(model_config, k, v)

    print(model_config)

    model = model_generator_llama(model_config)
    model.to(torch.bfloat16)
    print(model)

    saving_path = os.path.join(train_config.output_dir, 'last-ckpt.pt')
    states = model.state_dict()
    print(f'Saving ckpt at {saving_path}')
    torch.save(states, saving_path)

if __name__ == "__main__":
    fire.Fire(main)

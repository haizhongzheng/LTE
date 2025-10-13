# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

@dataclass
class train_config:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=2048
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=2e-5
    moe_lr: float=1e-2
    lr_scheduler: str='cosine'
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    save_model_epoch: bool = False
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    lte: bool=False
    moe_eta: float=0
    log_step: int=100
    c4_num: int=100000
    eval_mode: bool=False
    activation_save_dir: str=None


@dataclass
class model_config:
    ckpt_path: str=None
    eval: bool=False
    relu: bool=False
    kla: bool=False
    kla_sparsity: float=0
    lte: bool=False
    hard: bool=False
    moe_type: str=None
    moe_routing_mode: str=None
    moe_experts: int=0
    moe_experts_selected: int=0
    kmean_grouping: bool=True
    kmean_grouping_path: str=None
    use_pretrained: bool=False
    moe_rank: int=50
    keep_activation_output: bool=False
    dejavu: bool=False

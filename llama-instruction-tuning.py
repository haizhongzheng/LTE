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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
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
    evaluation,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from accelerate.utils import is_xpu_available

from lte_model_generator import model_generator_llama
import tulu_data

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, model_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    ################## Set distributed env  ##################
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print('local_rank', local_rank)
        if local_rank == 0:
            print(fsdp_config)
            print(train_config)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    ################## END - Set distributed env  ##################

    ################## Model & FSDP setting  ##################

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None

    for k, v in kwargs.items():
        setattr(model_config, k, v)

    model = model_generator_llama(model_config)

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.xpu.current_device() if is_xpu_available() else torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")


    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = train_config.context_length

    ################## END - Model & FSDP setting  ##################

    ################ LTE Setting ################
    model.config.lte = train_config.lte

    if model_config.lte:
        print('model_config.moe_type', model_config.moe_type)
        model.config.lte = model_config.lte
        model.config.hard = model_config.hard
        model.config.moe_routing_mode = model_config.moe_routing_mode
        model.config.kmean_group = model_config.kmean_grouping

        print('model.config.hard', model.config.hard)
        print('model_config.kmean_grouping', model_config.kmean_grouping)

        if model_config.moe_type == 'block': #Construct the moe routers
            if model_config.kmean_grouping:
                model.model.add_moe(moe_type=model_config.moe_type, experts=model_config.moe_experts, split_path=model_config.kmean_grouping_path, k=model_config.moe_experts_selected, hard=model_config.hard)
            else:
                model.model.add_moe(moe_type=model_config.moe_type, experts=model_config.moe_experts, split_path=None, k=None, hard=model_config.hard)

        if model_config.moe_type == 'row': #Construct the moe routers
            model.model.add_moe(moe_type=model_config.moe_type, rank=model_config.moe_rank, k=None)

        if not model_config.hard: # soft mode (phase 2)
            print('Load vanilla fine-tuned model!')
            print(f'Load model at {model_config.ckpt_path}')
            device = torch.device('cpu')
            if not model_config.use_pretrained:
                state_dict = torch.load(model_config.ckpt_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)

        else: # hard mode (phase 3)
            print('Load soft LTE model!')
            print(f'Load model at {model_config.ckpt_path}')
            device = torch.device('cpu')
            state_dict = torch.load(model_config.ckpt_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

            model.model.set_moe_hard()
            model.model.reset_moe_sparsity_statistics()

    ################ END - LTE Setting ################

    ################ Dataset Setting ################

    if (not train_config.enable_fsdp) or local_rank == 0:
        print(tokenizer)
        print(model_config)
        print(train_config)
        print(fsdp_config)

    train_dataset = tulu_data.get_tulu_data(model, tokenizer)
    train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    if (not train_config.enable_fsdp) or local_rank == 0:
        print(train_dl_kwargs)
        print(len(train_dataloader))

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    ## adjust model parameters

    #hard mode
    if model_config.lte and model_config.hard:
        for name, param in model.named_parameters():
            if 'moe' in name:
                param.requires_grad = False

    #soft mode
    if model_config.lte and not model_config.hard:
        for name, param in model.named_parameters():
            if 'moe' in name:
                param.requires_grad = True
            if 'moe.experts_masks' in name:
                param.requires_grad = False

    llm_params = [p for n, p in model.named_parameters() if (not ('moe' in n))]
    moe_params = [p for n, p in model.named_parameters() if ('moe' in n) and p.requires_grad]

    print('moe_params', len(moe_params))
    print('model.device', model.device)

    for n, p in  model.named_parameters():
        if 'moe' in n:
            p.data = p.data.to(torch.bfloat16)

    for n, p in  model.named_parameters():
        if 'moe' in n:
            p.data = p.data.to(model.device)

    if rank == 0:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n)

    if model_config.hard:
        moe_lr = 0
    else:
        moe_lr = train_config.moe_lr

    if train_config.use_peft:
        model.print_trainable_parameters()

    if (not train_config.enable_fsdp) or local_rank == 0:
        print('moe_lr', moe_lr)

    optimizer_grouped_parameters = [
        {'params': llm_params, "lr": train_config.lr, 'weight_decay': train_config.weight_decay},
        {"params": moe_params, "lr": moe_lr, "weight_decay": 0.1}
    ]

    # Initialize the optimizer and learning rate scheduler
    print('fsdp_config.optimizer', fsdp_config.optimizer)
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            optimizer_grouped_parameters
        )

    if train_config.lr_scheduler == 'step':
        print('step lr scheduler per epoch')
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    else:
        print('cosine lr scheduler per step')
        step_per_epoch = len(train_dataloader) // train_config.gradient_accumulation_steps
        total_steps = step_per_epoch * train_config.num_epochs
        print('total_steps for cosine lr:', total_steps)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    if train_config.eval_mode:
        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer)
        exit()

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        model_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

    if True:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        saving_path = os.path.join(train_config.output_dir, 'last-ckpt.pt')
        states = model.state_dict()
        if rank == 0:
            print(f'Saving ckpt at {saving_path}')
            torch.save(states, saving_path)

if __name__ == "__main__":
    fire.Fire(main)

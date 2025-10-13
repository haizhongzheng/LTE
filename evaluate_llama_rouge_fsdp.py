# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pkg_resources import packaging
import argparse

import fire
import random
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
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
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from datasets import load_metric
from evaluate import load

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
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

from llama_recipes.datasets import get_e2e_dataset

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
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)
from accelerate.utils import is_xpu_available

from lte_model_generator import model_generator_llama
from transformers import DataCollatorForSeq2Seq

from datasets import load_dataset
from datasets import load_metric
from evaluate import load

def get_xsum_val_dataloader(args, model, tokenizer):
    raw_datasets = load_dataset("xsum")

    prefix =  'Summarize this document:\n'
    postfix = '\n---\nSummary:\n'

    def preprocess_function(examples):
        # inputs_doc = [tokenizer.bos_token + prefix + doc for doc, summary in zip(examples["document"], examples["summary"])]
        inputs_doc = [prefix + doc for doc, summary in zip(examples["document"], examples["summary"])]
        inputs_summary = [postfix for doc, summary in zip(examples["document"], examples["summary"])]

        tokenized_doc = tokenizer(inputs_doc)
        tokenized_summary = tokenizer(inputs_summary, add_special_tokens=False)

        max_length = tokenizer.model_max_length - 100 # reserve for summary.
        example_num = len(tokenized_doc["input_ids"])

        rst = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(example_num):
            input_ids = tokenized_summary['input_ids'][i] #Reserve for the postfix
            max_rest_length = max_length - len(input_ids)

            input_ids = tokenized_doc['input_ids'][i][:max_rest_length] + input_ids

            rst['input_ids'].append(input_ids)
            rst['labels'].append(input_ids)

            attention_mask = [1] * len(input_ids)
            rst['attention_mask'].append(attention_mask)

        return rst

    val_dataset = raw_datasets['validation'].map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    cols_to_remove = val_dataset.column_names
    cols_to_remove.remove("input_ids")
    cols_to_remove.remove("attention_mask")
    cols_to_remove.remove("labels")
    val_dataset = val_dataset.remove_columns(cols_to_remove)

    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=data_collator, shuffle=False)

    val_dl_kwargs = get_dataloader_kwargs(args, val_dataset, tokenizer, "val")
    print(val_dl_kwargs)

    eval_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=args.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    return val_dataset, val_dataloader


def evaluate(args, raw_datasets, model, tokenizer):
    model.eval()

    val_dataset, val_dataloader = get_xsum_val_dataloader(args, model, tokenizer)
    prefix = [example["document"] for example in raw_datasets["validation"]]
    references = [example["summary"] for example in raw_datasets["validation"]]

    predictions = []

    if args.lte:
        model.model.reset_moe_sparsity_statistics()

    for inputs in tqdm(val_dataloader):
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=200)

        input_length = len(inputs['input_ids'][0])
        generated_texts = outputs[:, input_length:]
        # generated_texts = outputs
        generated_texts = [tokenizer.decode(example, skip_special_tokens=True) for example in generated_texts]
        # generated_texts = [tokenizer.decode(example, skip_special_tokens=False) for example in generated_texts]

        # print(input_length)
        # print(generated_texts)

        predictions = predictions + generated_texts

    # for inputs in tqdm(val_dataloader):
        # input_length = inputs['input_length']
        # del inputs['input_length']
        # if torch.cuda.is_available():
            # inputs = {key: value.cuda() for key, value in inputs.items()}
        # outputs = model.generate(**inputs, max_new_tokens=100,
            # pad_token_id=tokenizer.eos_token_id,
            # )

        # generated_texts_all = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        # generated_texts = [generated_text[input_length[i]:] for i, generated_text in enumerate(generated_texts_all)]

        # predictions = predictions + generated_texts

        if len(predictions) > 300:
            break

        if len(predictions) % 400 == 200:
            n = random.randint(0, len(predictions))
            print('***' * 50)
            print(f'Sampling n={n}')
            print("prefix[n]")
            print(prefix[n])
            print("predictions[n]")
            print(predictions[n])
            print("references[n]")
            print(references[n])

        # if args.lte:
        #     all_activations, sparse_activations = model.model.get_sparsity_statistics()
        #     print(f'all: {all_activations}\nsparse: {sparse_activations}')
        #     print(1.0 * sparse_activations / (all_activations + 0.1))

    # Calculate ROUGE score
    metric = load("rouge")
    results = metric.compute(predictions=predictions, references=references[:len(predictions)])
    print(results)

    n = random.randint(0, len(predictions))
    # for n in range(len(predictions)):
    print('***' * 50)
    print(f'Sampling n={n}')
    print("prefix[n]")
    print(prefix[n])
    print("predictions[n]")
    print(predictions[n])
    print("references[n]")
    print(references[n])

    rst={}
    rst['pred'] = predictions
    rst['prefix'] = prefix
    rst['ref'] = references

    return rst


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, model_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    else:
        torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

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
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        else:
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)



    # smoke_test = kwargs['smoke_test']
    # print(smoke_test)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        # model = LlamaForCausalLM.from_pretrained(
        #     train_config.model_name,
        #     load_in_8bit=True if train_config.quantization else None,
        #     device_map="auto" if train_config.quantization else None,
        #     use_cache=use_cache,
        # )
        # model.config.kla=False

        for k, v in kwargs.items():
            setattr(model_config, k, v)

        print(model_config)

        model = model_generator_llama(model_config)

    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

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


    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

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
        if is_xpu_available():
            model.to("xpu:0")
        else:
            model.to("cuda")


    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 2048

    print(model)


    raw_datasets = load_dataset("xsum")
    rst = evaluate(train_config, raw_datasets, model, tokenizer)


if __name__ == "__main__":
    import time
    start = time.time()
    fire.Fire(main)
    end = time.time()
    print('Script running time: ', end - start)
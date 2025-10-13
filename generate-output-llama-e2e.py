# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pkg_resources import packaging
import argparse

import fire
import random
import torch
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


def get_e2e_val_dataloader(tokenizer, model):
    raw_datasets = load_dataset("e2e_nlg")
    # raw_datasets = load_dataset("xsum")
    # raw_datasets = load_dataset("e2e_nlg_cleaned")

    prefix_input = 'Input:\n'
    prefix_output = '\n---\nOutput:\n'
    max_input_length = 1024
    max_target_length = 1024

    def preprocess_function(examples):
        inputs = [tokenizer.bos_token + prefix_input + doc + prefix_output for doc in examples["meaning_representation"]]
        rst = tokenizer(inputs, max_length=max_input_length, truncation=True)

        rst['input_length'] = []
        for example in inputs:
            rst['input_length'].append(len(example) - 2)

        return rst

    val_dataset = raw_datasets['validation'].map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    cols_to_remove = val_dataset.column_names
    cols_to_remove.remove("input_ids")
    cols_to_remove.remove("attention_mask")
    cols_to_remove.remove("input_length")
    val_dataset = val_dataset.remove_columns(cols_to_remove)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator, shuffle=False)

    return val_dataset, val_dataloader

def evaluate(args, raw_datasets, model, tokenizer):
    model.eval()

    val_dataset, val_dataloader = get_e2e_val_dataloader(tokenizer, model)
    prefix = [example["meaning_representation"] for example in raw_datasets["validation"]]
    references = [example["human_reference"] for example in raw_datasets["validation"]]

    predictions = []
    i = 0
    for inputs in tqdm((val_dataloader)):
        i += 1
        input_length = inputs['input_length']
        del inputs['input_length']
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                )

        generated_texts_all = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_texts = [generated_text[input_length[i]:] for i, generated_text in enumerate(generated_texts_all)]

        predictions = predictions + generated_texts

        if i % 10 == 1:
            n = random.randint(0, len(predictions) - 1)
            print(f'Sampling n={n}')
            print("prefix[n]")
            print(prefix[n])
            print("predictions[n]")
            print(predictions[n])
            print("references[n]")
            print(references[n])

    # Calculate ROUGE score
    metric = load("rouge")
    results = metric.compute(predictions=predictions, references=references[:len(predictions)])
    print(results)

    n = random.randint(0, len(predictions) - 1)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--output_dir", default='data-model/tmp', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="gpt", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--use_pretrained", action='store_true')

    ############## LTE related setting settings ##############
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--save_all_step_ckpt', action='store_true')
    parser.add_argument("--kla", action='store_true')
    parser.add_argument("--kla_sparsity", type=float, default=0)
    parser.add_argument("--relu", action='store_true')

    parser.add_argument("--kmean_grouping_path", type=str)
    parser.add_argument("--moe_experts_selected", type=int)

    ################ moe choice ################
    parser.add_argument("--lte", action='store_true')
    parser.add_argument("--hard", action='store_true')
    parser.add_argument("--moe_type", type=str)
    parser.add_argument("--moe_routing_mode", type=str)
    parser.add_argument("--moe_eta", type=float)
    parser.add_argument("--moe_experts", type=int, default=128)
    parser.add_argument("--kmean_grouping", action='store_true')

    ################ dejavu choice ################
    parser.add_argument("--dejavu", action='store_true')
    parser.add_argument("--dejavu_predictor_dir", type=str)

    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = 'left'

    print(args)

    model = model_generator_llama(args)

    ################ LTE Setting ################
    model.config.lte = args.lte

    if args.lte:
        print('args.moe_type', args.moe_type)
        model.config.lte = args.lte
        model.config.hard = args.hard
        model.config.moe_routing_mode = args.moe_routing_mode
        model.config.kmean_group = args.kmean_grouping

        print('model.config.hard', model.config.hard)
        print('args.kmean_grouping', args.kmean_grouping)

        if args.moe_type == 'block': #Construct the moe routers
            if args.kmean_grouping:
                model.model.add_moe(moe_type=args.moe_type, experts=args.moe_experts, split_path=args.kmean_grouping_path, k=args.moe_experts_selected, hard=args.hard)
            else:
                model.model.add_moe(moe_type=args.moe_type, experts=args.moe_experts, split_path=None, k=None, hard=args.hard)

        print(f'Load model at {args.ckpt_path}')
        device = torch.device('cpu')
        state_dict = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        model.model.set_moe_hard()
        model.model.reset_moe_sparsity_statistics()

    model.to(torch.bfloat16)
    model.cuda()

    # for n, p in model.named_parameters():
    #     if 'moe' in n and not 'experts_masks' in n:
    #         print(p)
    #         print(p.sum())

    raw_datasets = load_dataset("e2e_nlg")
    rst = evaluate(args, raw_datasets, model, tokenizer)
    torch.save(rst, 'data-data/tmp/e2e-rst.pt')
import torch
import numpy as np
import transformers
import copy
from typing import Optional, Dict, Sequence
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)
import evaluate
from dataclasses import dataclass, field
import argparse
from torch.nn.utils.rnn import pad_sequence

from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.configs import model_config as MODEL_CONFIG
from lte_model_generator import model_generator_llama
from llama_recipes.utils.eval_utils import mmlu_eval


parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--ckpt_path', type=str, help='path to the model checkpoint')

########## model setting ##########
parser.add_argument('--kla', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--relu', action='store_true')
parser.add_argument('--use_pretrained', action='store_true')

parser.add_argument('--lte', action='store_true')
parser.add_argument('--hard', action='store_true')
parser.add_argument('--moe_routing_mode', type=str)
parser.add_argument('--moe_type', type=str)
parser.add_argument('--moe_experts', type=int)
parser.add_argument('--moe_rank', type=int, default=50)

parser.add_argument('--kmean_grouping', action='store_true')
parser.add_argument('--kmean_grouping_path', type=str)

args = parser.parse_args()

assert (not args.use_pretrained) or (args.ckpt_path is None), "(not args.use_pretrained) or (args.ckpt_path is None) is False."


# print(training_args)
# print(extra_args)

################ Model Setting ################
model_name = args.model_name

# model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model_generator_llama(args)
print(model_name)
print(model.config)
# print(tokenizer)
# print(tokenizer.pad_token_id)

# import ipdb; ipdb.set_trace()

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
            model.model.add_moe(moe_type=args.moe_type, experts=args.moe_experts, split_path=args.kmean_grouping_path, k=None, hard=args.hard)
        else:
            model.model.add_moe(moe_type=args.moe_type, experts=args.moe_experts, split_path=None, k=None, hard=args.hard)

    if args.moe_type == 'row': #Construct the moe routers
        model.model.add_moe(moe_type=args.moe_type, rank=args.moe_rank, k=None)

    model.model.set_moe_hard()
    model.model.reset_moe_sparsity_statistics()

################ END - LTE Setting ################
if args.ckpt_path is not None:
    print(f'load model ckpt at {args.ckpt_path}')
    device = torch.device('cpu')
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    if args.lte:
        model.model.set_moe_hard()
        model.model.reset_moe_sparsity_statistics()

model = model.cuda()
model.to(torch.bfloat16)

################ Model Setting end ################

mmlu_eval(args, model)
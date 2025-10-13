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
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, collate_fn=data_collator, shuffle=False)

    return val_dataset, val_dataloader


def evaluate(args, raw_datasets, model, tokenizer):
    model.eval()

    val_dataset, val_dataloader = get_xsum_val_dataloader(args, model, tokenizer)
    prefix = [example["document"] for example in raw_datasets["validation"]]
    references = [example["summary"] for example in raw_datasets["validation"]]

    predictions = []

    if args.lte:
        model.model.reset_moe_sparsity_statistics()

    count = 0
    for inputs in tqdm(val_dataloader):
        count += 1
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(max_new_tokens=100, **inputs)

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

        # if len(predictions) > 300:
        #     break

        if len(predictions) % 200 == 100:
            print('len(predictions):', len(predictions))
            n = random.randint(0, len(predictions) - 1)
            print('***' * 50)
            print(f'Sampling n={n}')
            print("prefix[n]")
            print(prefix[n])
            print("predictions[n]")
            print(predictions[n])
            print("references[n]")
            print(references[n])

            if args.lte:
                all_activations, sparse_activations = model.model.get_sparsity_statistics()
                print(f'all: {all_activations}\nsparse: {sparse_activations}')
                print(1.0 * sparse_activations / (all_activations + 0.1))

    # Calculate ROUGE score
    metric = load("rouge")
    results = metric.compute(predictions=predictions, references=references[:len(predictions)])
    print(results)

    if args.lte:
        all_activations, sparse_activations = model.model.get_sparsity_statistics()
        print(f'all: {all_activations}\nsparse: {sparse_activations}')
        print(1.0 * sparse_activations / (all_activations + 0.1))

    n = random.randint(0, len(predictions) - 1)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--output_dir", default='data-model/tmp', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="gpt", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--model_name", default="SparseLLM/ReluLLaMA-7B", type=str,
                        help="The model architecture to be fine-tuned.")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument('--use_pretrained', action='store_true')

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

        if not args.hard: # soft mode (phase 2)
            print('Load vanilla fine-tuned model!')
            print(f'Load model at {args.ckpt_path}')
            device = torch.device('cpu')
            state_dict = torch.load(args.ckpt_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

        else: # hard mode (phase 3)
            print('Load soft LTE model!')
            print(f'Load model at {args.ckpt_path}')
            device = torch.device('cpu')
            state_dict = torch.load(args.ckpt_path, map_location=device)
            model.load_state_dict(state_dict)

            model.model.set_moe_hard()
            model.model.reset_moe_sparsity_statistics()

            for name, param in model.named_parameters():
                if 'moe' in name:
                    param.requires_grad = False

    model.cuda()
    model.eval()

    model.to(torch.bfloat16)

    # prefix =  'Summarize this document:\n'
    # postfix = '\n---\nSummary:\n'

    # input_text = """The Association of School and College Leaders says England\'s schools have had to make more than £1bn savings this year, rising to £3bn by 2020.\nThe government says school funding is at a record £40bn, with rises ahead.\nEducation Secretary Justine Greening will hear heads\' cash grievances at Friday\'s ASCL conference in Birmingham.\nShe is due to address the union, which has published a survey of its members on the issue.\nIt suggests schools are finding it difficult to make savings without cutting provision and that things are predicted to get worse over the next two years.\nCost pressures are rising as greater pay, pension and national insurance costs are having to be covered from school budgets.\nASCL complains a new funding formula for schools has reduced the basic level of school funding going forwards by too much.\nThe meeting comes two days after requests for more money to spend on daily school costs were ignored by the chancellor in the Budget.\nPhilip Hammond however did pledge £500m for school buildings, mainly new free schools - some of which could be grammar schools.\nOne respondent said his school was moving to a "bare bones education", in which "the components that make education special and enjoyable are being eroded away".\nSome 95% of the 1,054 heads, deputies and senior teachers responding to the survey said they had cut back on support services - including equipment and materials, as well as mental health and special needs support.\nMore than eight out of 10 said class sizes had increased - a claim strongly refuted by the Department for Education.\nAnd more than two-thirds said they had cut back on activities like clubs and trips.\nJust under three-quarters of respondents with GCSE-level classes said they had cut courses and just over three-quarters of heads with A-level students said they had also reduced subjects.\nForeign modern languages, music, arts and drama were among subjects removed at A-level.\nAnother said: "Through no fault of their own, students will have restricted subject choices, in larger class sizes with less pastoral support, whilst still being expected to perform at the highest of standards - nonsense!"\nOne head said his school may have to axe its sixth form provision for next year and another said his school was starting to "creak" with all staff working to full capacity.\nInterim general secretary, Malcolm Trobe, said: "School leaders will do their utmost to protect provision, as they always do, but they cannot provide everything that is asked of them without the resources they need.\n"Unless the government invests more in the education system, there will be a significant impact on the lives and life chances of young people."\nA spokesman for the DfE said: "As this week\'s Budget demonstrates, the government is determined to ensure every child has access to a good school place and is given the opportunity to fulfil their potential.\n"The government has protected the core schools budget in real terms since 2010, with school funding at its highest level on record at more than £40bn in 2016-17 - and that is set to rise, as pupil numbers rise over the next two years, to £42bn by 2019-20."""

    # input_text = prefix + input_text + postfix

    # tokenized_doc = tokenizer(input_text, return_tensors='pt')

    # if torch.cuda.is_available():
    #     inputs = {key: value.cuda() for key, value in tokenized_doc.items()}
    # outputs = model.generate(**inputs, max_new_tokens=200)

    # outputs = outputs.to('cpu')
    # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(decoded_output)

    raw_datasets = load_dataset("xsum")

    import time
    start = time.time()
    rst = evaluate(args, raw_datasets, model, tokenizer)
    end = time.time()
    print('Evaluation time: ', end - start)
"""
Script for preparing the Tulu V2 data for fine-tuning an OLMo model.
"""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import DataLoader

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import json
import io
from rich.progress import track

# from olmo.tokenizer import Tokenizer
# from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)

def filter(example):
    return example["n_labels"] > 0

# def main(opts) -> None:
def get_tulu_data(model, tokenizer, num=None):
    # dataset = ds.load_dataset("allenai/tulu-v2-sft-mixture", split="train[:2000]")
    if num is not None:
        dataset = ds.load_dataset("allenai/tulu-v2-sft-mixture", split=f"train[:{num}]")
    else:
        dataset = ds.load_dataset("allenai/tulu-v2-sft-mixture", split="train")

    log.info("Tokenizing dataset...")
    # import ipdb; ipdb.set_trace()

    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=tokenizer.model_max_length),
        batched=False,
        remove_columns=["dataset", "id", "messages"],
        num_proc=1,
    )

    # import ipdb; ipdb.set_trace()

    print("Filtering dataset...")
    n = len(dataset)
    dataset = dataset.filter(filter, batched=False, num_proc=1)
    print(f"Filtered out {n - len(dataset):,d} examples")

    # print("Counting tokens...")
    # total_tokens = 0
    # for ex in track(dataset):
    #     assert len(ex["input_ids"]) == tokenizer.model_max_length
    #     total_tokens += len(ex["input_ids"])
    # print(f"Total tokens: {total_tokens:,d}")

    # import ipdb; ipdb.set_trace()

    print('done!')

    dataset = dataset.remove_columns("n_labels")

    return dataset

# all example will be padded to the same length
def preprocess(example, tokenizer, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
            label_mask += [True] * len(content_tokens)
            # mask out the last '\n'
            assert content_tokens[-2] == tokenizer.eos_token_id or content_tokens[-3] == tokenizer.eos_token_id
            label_mask[-1] = False
            if content_tokens[-2] != tokenizer.eos_token_id:
                label_mask[-2] = False
        else:
            content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    labels = [value if label else -100 for value, label in zip(input_ids, label_mask)]

    # print(1111)

    return {"input_ids": input_ids, "labels": labels, "n_labels": n_labels}
    # return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}

def adjust_tokenizer(model, tokenizer):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    print('Set up tokenizer for alpaca finetuning...')
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    model_name = 'meta-llama/Llama-2-7b-hf'
    model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=2048)

    print(tokenizer)

    adjust_tokenizer(model, tokenizer)

    print(tokenizer)
    get_tulu_data(model, tokenizer)


# if __name__ == "__main__":
#     prepare_cli_environment()
#     opts = get_parser().parse_args()
#     main(opts)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name = 'meta-llama/Llama-2-7b-chat-hf'
# tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=2048)
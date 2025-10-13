# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_xsum_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("xsum", split=split)

    prompt = (
        f"Summarize this document:\n{{document}}\n---\nSummary:\n"
    )

    prefix = 'Summarize this document:\n'
    postfix = '\n---\nSummary:\n'

    def apply_prompt_template(sample):
        return {
            "prompt": sample["document"],
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        max_length = 2048

        tokenized_prefix = tokenizer.encode(tokenizer.bos_token + prefix, add_special_tokens=False)
        tokenized_postfix = tokenizer.encode(postfix, add_special_tokens=False)

        prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        prompt_max_length = max_length - len(summary) - len(tokenized_prefix) - len(tokenized_postfix)
        prompt = prompt[:prompt_max_length]

        sample = {
            "input_ids": tokenized_prefix + prompt + tokenized_postfix + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary) + len(tokenized_prefix) + len(tokenized_postfix)),
            "labels": [-100] * len(tokenized_prefix + prompt + tokenized_postfix) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

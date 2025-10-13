# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_wiki_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("wikitext", "wikitext-103-v1", split=split)

    def apply_prompt_template(sample):
        return {
            "prompt": sample["text"]
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    max_length = 2048

    def tokenize_data(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"] + tokenizer.eos_token, add_special_tokens=False)
        if len(prompt) > 2048:
            prompt = prompt[:2047] + prompt[-1:]

        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "labels": prompt,
            }

        return sample

    dataset = dataset.map(tokenize_data, remove_columns=list(dataset.features))

    return dataset

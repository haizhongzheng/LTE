# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_e2e_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("e2e_nlg", split=split)

    prompt = (
        f"Input:\n{{meaning_representation}}\n---\nOutput:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(meaning_representation=sample["meaning_representation"]),
            "human_reference": sample["human_reference"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["human_reference"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
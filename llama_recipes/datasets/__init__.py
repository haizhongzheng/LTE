# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama_recipes.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset

from llama_recipes.datasets.e2e_dataset import get_e2e_dataset
from llama_recipes.datasets.xsum_dataset import get_xsum_dataset
from llama_recipes.datasets.wiki_dataset import get_wiki_dataset
from llama_recipes.datasets.c4_dataset import get_c4_dataset
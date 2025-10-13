# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
import datasets

dataset = datasets.load_dataset("c4", "en", split=f"train")
dataset = datasets.load_dataset("c4", "en", split=f"validation")


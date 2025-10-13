from ast import arg
from re import template
import os
import km_utils as utils
import numpy as np
import torch
import argparse
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, help='path to the model checkpoint')
parser.add_argument('--res_path', type=str, default='data-data/tmp', help='path to store the results of moefication')
parser.add_argument('--saving_path', type=str)
parser.add_argument('--num-layer', type=int, default=12, help='number of layers')
parser.add_argument('--num-expert', type=int, default=96, help='number of experts')
# parser.add_argument('--templates', type=str, default='roberta.encoder.layer.{}.intermediate.dense.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')

parser.add_argument('--model_type', type=str, default='llama')

args = parser.parse_args()

if args.model_type == 'roberta':
    templates = 'roberta.encoder.layer.{}.intermediate.dense.weight'
if args.model_type == 'gpt2':
    templates = 'transformer.h.{}.mlp.c_fc.weight'
if args.model_type == 'llama':
    templates = 'model.layers.{}.mlp.gate_proj.weight'

if not os.path.exists(args.res_path):
    os.makedirs(args.res_path)

config = utils.ModelConfig(args.model_path, args.res_path, split_num=args.num_expert)

templates = templates.split(',')

grouping_rst = []
for template in templates:
    for i in tqdm.tqdm(range(args.num_layer)):
        split = utils.ParamSplit(config, template, i, model_type=args.model_type)
        split.split()
        split.cnt()
        grouping_rst.append(split.labels)

directory = os.path.dirname(args.saving_path)
if not os.path.exists(directory):
    os.makedirs(directory)

torch.save(grouping_rst, args.saving_path)
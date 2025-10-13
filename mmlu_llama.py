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
parser.add_argument('--moe_experts_selected', type=int)
parser.add_argument('--moe_rank', type=int, default=50)

parser.add_argument('--kmean_grouping', action='store_true')
parser.add_argument('--kmean_grouping_path', type=str)

################ dejavu choice ################
parser.add_argument("--dejavu", action='store_true')
parser.add_argument("--dejavu_predictor_dir", type=str)

parser.add_argument('--rst_file', type=str)


args = parser.parse_args()

assert (not args.use_pretrained) or (args.ckpt_path is None), "(not args.use_pretrained) or (args.ckpt_path is None) is False."

IGNORE_INDEX=-100

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": 'The eval batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

mmlu_dataset = load_dataset("json", data_files={
    'eval': 'data/mmlu/five_shot_mmlu_val.json',
    'test': 'data/mmlu/five_shot_mmlu_test.json',
})

hfparser = transformers.HfArgumentParser((
        TrainingArguments
    ))
training_args, extra_args = \
    hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

print(training_args)

print(extra_args)

# print(mmlu_dataset)

################ MMLU Setting ################
mmlu_split = 'eval'
max_mmlu_samples = 200
mmlu_dataset = mmlu_dataset[mmlu_split]
# mmlu_dataset = mmlu_dataset.select(range(max_mmlu_samples))

print(mmlu_dataset)
################ MMLU Setting end ################


################ Model Setting ################
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.pad_token_id = 2

# model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model_generator_llama(args)
print(model_name)
print(model.config)
print(tokenizer)
print(tokenizer.pad_token_id)

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

abcd_idx = [
    tokenizer("A", add_special_tokens=False).input_ids[0],
    tokenizer("B", add_special_tokens=False).input_ids[0],
    tokenizer("C", add_special_tokens=False).input_ids[0],
    tokenizer("D", add_special_tokens=False).input_ids[0],
]
accuracy = evaluate.load("accuracy")
print(abcd_idx)

data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=2048,
        target_max_len=16,
        train_on_source=False,
        predict_with_generate=False,
    )

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator
)
print(trainer)

data_loader = trainer.get_eval_dataloader(mmlu_dataset)

print(trainer.data_collator.source_max_len)
print('len(data_loader)', len(data_loader))

# import ipdb; ipdb.set_trace()

model.eval()
preds, refs = [], []
loss_mmlu = 0

for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
    # if batch_idx > 100:
    #     break
    (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
    # There are two tokens, the output, and eos token.
    # for i, logit in enumerate(logits):
    #     label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
    #     logit_abcd = logit[label_non_zero_id-1][abcd_idx]
    #     preds.append(torch.argmax(logit_abcd).item())

    for i, logit in enumerate(logits):
        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
        preds.append(torch.argmax(logit_abcd).item())

    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
    refs += [abcd_idx.index(label) for label in labels.tolist()]
    loss_mmlu += loss.item()

print(labels)
print(refs)
print(preds)

results = {'mmlu_loss':loss_mmlu/len(data_loader)}
subject = mmlu_dataset['subject']
subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
for s,p,r in zip(subject, preds, refs):
    subjects[s]['preds'].append(p)
    subjects[s]['refs'].append(r)
subject_scores = []
for subject in subjects:
    subject_score = accuracy.compute(
        references=subjects[subject]['refs'],
        predictions=subjects[subject]['preds']
    )['accuracy']
    results[f'mmlu_accuracy_{subject}'] = subject_score
    subject_scores.append(subject_score)
results[f'mmlu_accuracy'] = np.mean(subject_scores)
    # trainer.log(results)
print(results)

print('mmlu_accuracy:', results['mmlu_accuracy'])

if args.rst_file is not None:
    f = open(args.rst_file, "a")
    f.write(f"mmlu {results['mmlu_accuracy']}\n")
    f.close()

if args.lte:
    all_activations, sparse_activations = model.model.get_sparsity_statistics()
    print('all_activations, sparse_activations', all_activations, sparse_activations)
    print('sparse ratio:', sparse_activations / all_activations)
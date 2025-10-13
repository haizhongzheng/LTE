# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime
from datetime import timedelta

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.eval_utils import mmlu_eval
from accelerate.utils import is_xpu_available, is_ccl_available

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, model_config, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_eval_ppl = float("inf")

    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        if is_xpu_available():
                            batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                        else:
                            batch[key] = batch[key].to(local_rank)
                    else:

                        if is_xpu_available():
                            batch[key] = batch[key].to('xpu:0')
                        else:
                            batch[key] = batch[key].to('cuda:0')
                with autocast():
                    if train_config.lte:
                        output = model(**batch)
                        loss_generation, moe_mask_list = output[0].loss, output[1]
                        loss = loss_generation
                        if model_config.lte and model_config.moe_routing_mode == 'sigmoid' and not model_config.hard:
                            loss_efficiency = 0
                            loss_separability = 0
                            # import ipdb; ipdb.set_trace()
                            for moe_mask in moe_mask_list:
                                if 'attention_mask'  in batch:
                                    loss_efficiency += ((moe_mask * moe_mask).mean(dim=-1) * batch['attention_mask']).sum() / batch['attention_mask'].sum()
                                    loss_separability += ((1 / ((moe_mask - 0.5) * (moe_mask - 0.5) + 1)).mean(dim=-1) * batch['attention_mask']).sum() / batch['attention_mask'].sum()
                                else: # TULU dataset
                                    loss_efficiency += ((moe_mask * moe_mask).mean(dim=-1) * (batch['labels'] != -100)).sum() / (batch['labels'] != -100).sum()
                                    loss_separability += ((1 / ((moe_mask - 0.5) * (moe_mask - 0.5) + 1)).mean(dim=-1) * (batch['labels'] != -100)).sum() / (batch['labels'] != -100).sum()

                            loss_efficiency = loss_efficiency / len(moe_mask_list)
                            loss_separability = loss_separability / len(moe_mask_list)

                            loss += train_config.moe_eta * loss_efficiency
                            loss += 0.5 * loss_separability
                    else:
                        loss = model(**batch).loss
                        loss_generation = loss

                    ## log during training
                    if step % train_config.log_step == 1 and ((not train_config.enable_fsdp) or local_rank == 0):
                        print('Step: ', step)
                        print('loss_generation:', loss_generation.item())
                        if train_config.lte and not model_config.hard:
                            print('loss_efficiency:', loss_efficiency.item())
                            print('loss_separability:', loss_separability.item())
                        if train_config.lte:
                            moe_mask_list = [i.detach().cpu() for i in moe_mask_list]
                            moe_mask_all = torch.cat(moe_mask_list, 0)
                            # print('Mean:', moe_mask_all.mean().item())
                            # moe_mask_all = moe_mask_all.detach().cpu()
                            print(f'>=0.5: {((moe_mask_all >= 0.5).sum() / (moe_mask_all >= 0).sum()).item()}')
                            print('====' * 5)

                loss = loss / gradient_accumulation_steps
                if train_config.save_metrics:
                    train_step_loss.append(loss.detach().float().item())
                    train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            scaler.unscale_(optimizer)
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        # pbar.update(1)
                        if train_config.lr_scheduler == 'cosine':
                            lr_scheduler.step()
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        optimizer.step()
                        optimizer.zero_grad()
                        # pbar.update(1)
                        if train_config.lr_scheduler == 'cosine':
                            lr_scheduler.step()

                # pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                if train_config.save_metrics:
                    save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

                # end of the batch loop
                # if step > 50: # test pipeline
                #     break

            pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if train_config.enable_fsdp:
            if rank==0:
                if is_xpu_available():
                    print(f"Max XPU memory allocated was {memtrace.peak} GB")
                    print(f"Max XPU memory reserved was {memtrace.max_reserved} GB")
                    print(f"Peak active XPU memory was {memtrace.peak_active_gb} GB")
                    print(f"Xpu Malloc retires : {memtrace.xpu_malloc_retires}")
                else:
                    print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                    print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                    print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                    print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            if is_xpu_available():
                print(f"Max XPU memory allocated was {memtrace.peak} GB")
                print(f"Max XPU memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active XPU memory was {memtrace.peak_active_gb} GB")
                print(f"Xpu Malloc retires : {memtrace.xpu_malloc_retires}")
            else:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        # Update the learning rate as needed
        if train_config.lr_scheduler == 'step':
            lr_scheduler.step()

        if train_config.enable_fsdp:
            dist.barrier()
        if not train_config.run_validation and train_config.save_model:
            if not os.path.exists(train_config.output_dir):
                os.makedirs(train_config.output_dir, exist_ok=True)
            saving_path = os.path.join(train_config.output_dir, 'last-ckpt.pt')
            states = model.state_dict()
            if rank == 0: # for mmlu
                print(f'Saving ckpt at {saving_path}')
                torch.save(states, saving_path)
                if train_config.save_model_epoch:
                    saving_path = os.path.join(train_config.output_dir, f'epoch_{epoch+1}-ckpt.pt')
                    print(f'Saving ckpt at {saving_path}')
                    torch.save(states, saving_path)
                # mmlu_eval(model_config, model)

        if train_config.run_validation:
            if train_config.enable_fsdp:
                dist.barrier()
                if not os.path.exists(train_config.output_dir):
                    os.makedirs(train_config.output_dir, exist_ok=True)
                saving_path = os.path.join(train_config.output_dir, 'last-ckpt.pt')
                states = model.state_dict()
                if rank == 0:
                    print(f'Saving ckpt at {saving_path}')
                    torch.save(states, saving_path)

            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    dist.barrier()
                    if not os.path.exists(train_config.output_dir):
                        os.makedirs(train_config.output_dir, exist_ok=True)
                    saving_path = os.path.join(train_config.output_dir, 'last-ckpt.pt')
                    states = model.state_dict()
                    if rank == 0:
                        print(f'Saving ckpt at {saving_path}')
                        torch.save(states, saving_path)

                        if best_eval_ppl > eval_ppl:
                            saving_path = os.path.join(train_config.output_dir, 'best-ckpt.pt')
                            best_eval_ppl = eval_ppl
                            print('best ppl: ', best_eval_ppl)
                            if train_config.dataset == 'c4_dataset':
                                print(f'Saving ckpt at {saving_path}')
                                torch.save(states, saving_path)


            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    if len(epoch_times) > 0:
        avg_epoch_time = sum(epoch_times)/ len(epoch_times)
        avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
        avg_train_prep = sum(train_prep)/len(train_prep)
        avg_train_loss = sum(train_loss)/len(train_loss)
        if train_config.run_validation:
            avg_eval_prep = sum(val_prep)/len(val_prep)
            avg_eval_loss = sum(val_loss)/len(val_loss)

        results['avg_train_prep'] = avg_train_prep
        results['avg_train_loss'] = avg_train_loss
        if train_config.run_validation:
            results['avg_eval_prep'] = avg_eval_prep
            results['avg_eval_loss'] = avg_eval_loss
        results["avg_epoch_time"] = avg_epoch_time
        results["avg_checkpoint_time"] = avg_checkpoint_time
        if train_config.save_metrics:
            results["metrics_filename"] = metrics_filename

    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results

def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    activation_data_list = []
    with MemoryTrace() as memtrace:
        if train_config.lte:
            model.model.reset_moe_sparsity_statistics()
            all_activations, sparse_activations = model.model.get_sparsity_statistics()
            print(f'all: {all_activations}\nsparse: {sparse_activations}')

        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                # import ipdb; ipdb.set_trace()

                if model.config.keep_activation_output:
                    mlp_actication = model.model.get_mlp_activation_output_list()
                    activation_data_list.append(mlp_actication)
                    if len(activation_data_list) > 50 and local_rank == 0:
                        activation_save_dir = train_config.activation_save_dir
                        if not os.path.exists(activation_save_dir):
                            os.makedirs(activation_save_dir, exist_ok=True)
                        for i in range(len(activation_data_list[0])):
                            input = [activation_data_list[j][i][0] for j in range(len(activation_data_list))]
                            activation_data = [activation_data_list[j][i][1] for j in range(len(activation_data_list))]
                            print('saving activation data to ', f'{activation_save_dir}/activation_layer{i}.pt')
                            torch.save({'input': input,
                                        'activation': activation_data}, f'{activation_save_dir}/activation_layer{i}.pt')
                        exit()

                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

            if train_config.lte:
                all_activations, sparse_activations = model.model.get_sparsity_statistics()
                if all_activations > 0:
                    print(f'all: {all_activations}\nsparse: {sparse_activations}')
                    print(1.0 * sparse_activations / all_activations)

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=90) )


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""


    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
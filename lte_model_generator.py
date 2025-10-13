import torch
import os

from transformers import AutoConfig
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)

def model_generator_llama(args):
    if args.model_name in ["SparseLLM/ReluLLaMA-7B", "meta-llama/Llama-2-7b-hf", "huggyllama/llama-7b"]:
        model_name = args.model_name
    else:
        raise "Model name not found."

    print('model_name', model_name)

    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    # print(config)

    if args.relu:
        print('Use Relu as the activation functinon!!!')
        config.hidden_act = "relu"

    config.use_cache = False

    if args.eval:
        config.use_cache = True

    if hasattr(args, 'keep_activation_output'):
        config.keep_activation_output = args.keep_activation_output
    else:
        config.keep_activation_output = False

    model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=None,
            device_map=None,
            config=config
        )

    ## KLA sparsity
    model.config.kla = args.kla
    if args.kla:
        model.config.kla_sparsity = args.kla_sparsity

    if args.use_pretrained:
        print('Use the pretrained model in huggingface!!')
        return model

    if not args.lte and args.ckpt_path:
        device = torch.device('cpu')
        print(f'Load model at {args.ckpt_path}')
        state_dict = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    if hasattr(args, 'dejavu') and args.dejavu:
        model.model.add_dejavu(args.dejavu_predictor_dir)

    # user_input = input("Please enter something: ")
    # # Print what the user entered
    # print("You entered:", user_input)

    # ################ LTE Setting ################
    # model.config.lte = args.lte

    # if args.lte:
    #     print('args.moe_type', args.moe_type)
    #     model.config.lte = args.lte
    #     model.config.hard = args.hard
    #     model.config.moe_routing_mode = args.moe_routing_mode
    #     model.config.kmean_group = args.kmean_grouping

    #     print('model.config.hard', model.config.hard)
    #     print('args.kmean_grouping', args.kmean_grouping)

    #     if args.moe_type == 'block': #Construct the moe routers
    #         if args.kmean_grouping:
    #             model.model.add_moe(moe_type=args.moe_type, experts=args.moe_experts, split_path=args.kmean_grouping_path, k=args.moe_experts_selected, hard=args.hard)
    #         else:
    #             model.model.add_moe(moe_type=args.moe_type, experts=args.moe_experts, split_path=None, k=None, hard=args.hard)

    #     if not args.hard: # soft mode (phase 2)
    #         print('Load vanilla fine-tuned model!')
    #         print(f'Load model at {args.ckpt_path}')
    #         device = torch.device('cpu')
    #         state_dict = torch.load(args.ckpt_path, map_location=device)
    #         model.load_state_dict(state_dict, strict=False)

    #     else: # hard mode (phase 3)
    #         print('Load soft LTE model!')
    #         print(f'Load model at {args.ckpt_path}')
    #         device = torch.device('cpu')
    #         state_dict = torch.load(args.ckpt_path, map_location=device)
    #         model.load_state_dict(state_dict)

    #         model.model.set_moe_hard()
    #         model.model.reset_moe_sparsity_statistics()

            # for name, param in model.named_parameters():
            #     if 'moe' in name:
            #         param.requires_grad = False

    # user_input = input("Please enter something: ")
    # # Print what the user entered
    # print("You entered:", user_input)

    return model
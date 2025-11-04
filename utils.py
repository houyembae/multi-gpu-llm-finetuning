import os
from typing import Tuple, Optional
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from unsloth import FastLanguageModel


def create_and_prepare_model(model_args, data_args, training_args):
    bnb_config = None
    if model_args.use_4bit_quantization or model_args.use_8bit_quantization:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            load_in_8bit=model_args.use_8bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.use_unsloth:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=model_args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=model_args.use_4bit_quantization,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=model_args.use_flash_attn,
            torch_dtype=torch.float16,
        )

    if model_args.use_4bit_quantization or model_args.use_8bit_quantization:
        model = prepare_model_for_kbit_training(model)

    peft_config = None
    if model_args.use_peft_lora:
        target_modules = model_args.lora_target_modules.split(",") if model_args.lora_target_modules else []
        
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, peft_config, tokenizer


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=True):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"text": batch}

    raw_datasets = DatasetDict()
    
    for split in data_args.splits.split(","):
        try:
            dataset = load_dataset(data_args.dataset_name, split=split)
        except:
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type ({split}) not recognized.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def print_gpu_memory_usage():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
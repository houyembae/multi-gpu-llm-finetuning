import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import SFTTrainer
from utils import create_and_prepare_model, create_datasets


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={"help": "chatml|zephyr|none. Pass 'none' if dataset is already formatted."}
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules for LoRA"}
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"}
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"}
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"}
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables flash attention for training"}
    )
    use_peft_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables PEFT LoRA for training"}
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit"}
    )
    use_4bit_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables loading model in 4bit"}
    )
    use_reentrant: Optional[bool] = field(
        default=True,
        metadata={"help": "Gradient Checkpointing parameter"}
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training"}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="owl-health/medical-dialogue-to-soap-summary",
        metadata={"help": "The preference dataset to use."}
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."}
    )
    dataset_text_field: str = field(
        default="text", 
        metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=2048)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends eos_token_id at the end of each sample."}
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens."}
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separated list of splits to use from dataset."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    train_dataset, eval_dataset = create_datasets(
        tokenizer, 
        data_args,
        training_args
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=data_args.packing,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
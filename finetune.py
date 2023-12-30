### Tutorial: https://www.labellerr.com/blog/hands-on-with-fine-tuning-llm/


import argparse
from pathlib import Path
import logging
import torch

from datasets import (
    load_dataset,
    Dataset,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
)

from prompt import make_prompt


def load_data(datapath: Path) -> Dataset:
    logging.info('Loading the dataset')
    dataset = load_dataset('csv', data_files=str(datapath), split="train")
    return dataset


def load_pretrained_model():
    logging.info('Loading the pretrained model')
    model_name = 'tiiuae/falcon-7b'

    # Configuring the BitsAndBytes quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # Disabling cache usage in the model configuration
    model.config.use_cache = False

    tokeniser = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    tokeniser.pad_token = tokeniser.eos_token

    return model, tokeniser


def finetune(data: Dataset, model: AutoModelForCausalLM, tokeniser: AutoTokenizer):
    logging.info('Training the model')
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]
    )

    # Define the directory to save training results
    output_dir = "./results"

    # Set the batch size per device during training
    per_device_train_batch_size = 4

    # Number of steps to accumulate gradients before updating the model
    gradient_accumulation_steps = 4

    # Choose the optimizer type (e.g., "paged_adamw_32bit")
    optim = "paged_adamw_32bit"

    # Interval to save model checkpoints (every 10 steps)
    save_steps = 100

    # Interval to log training metrics (every 10 steps)
    logging_steps = 10

    # Learning rate for optimization
    learning_rate = 2e-4

    # Maximum gradient norm for gradient clipping
    max_grad_norm = 0.3

    # Maximum number of training steps
    max_steps = 50

    # Warmup ratio for learning rate scheduling
    warmup_ratio = 0.03

    # Type of learning rate scheduler (e.g., "constant")
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,  # Use mixed precision training (16-bit)
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
    )

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="###INPUT:",
        response_template="###OUTPUT",
        tokenizer=tokeniser,
        mlm=False,
    )

    # Set the maximum sequence length
    max_seq_length = 512

    # Create a trainer instance using SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokeniser,
        args=training_arguments,
        formatting_func=make_prompt,
        data_collator=collator,
    )

    # Iterate through the named modules of the trainer's model
    for name, module in trainer.model.named_modules():

        # Check if the name contains "norm"
        if "norm" in name:
            # Convert the module to use torch.float32 data type
            module = module.to(torch.float32)

    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True, help='Path to data to finetune from')
    args = parser.parse_args()

    data = load_data(args.data)
    model, tokeniser = load_pretrained_model()
    finetune(data, model, tokeniser)
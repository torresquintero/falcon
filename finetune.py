import argparse
from pathlib import Path
import logging
import pandas as pd

from datasets import (
    load_dataset,
    Dataset,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from prompt import make_prompt


def load_data(datapath: Path) -> Dataset:
    logging.info('Loading the dataset')
    dataset = load_dataset('csv', data_files=str(datapath), split="train")
    return dataset


def load_pretrained_model():
    logging.info('Loading the pretrained model')
    model_name = 'tiiuae/falcon-7b'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )

    tokeniser = AutoTokenizer.from_pretrained(model_name)

    tokeniser.pad_token = tokeniser.eos_token

    return model, tokeniser


def tokenise_data(data: Dataset, tokeniser: AutoTokenizer):
    data = data.map(gen_and_tok_prompt)


    full_input = gen_prompt(text_input)
    tok_full_prompt = tokenizer(full_input, padding = True , truncation =True)
    return tok_full_prompt


    # ALE YOU WERE IN THE MIDDLE OF GETTING THE MAP WORKING FOR THE DATASET INTO THE PROMPT
    # IN PARTICULAR FIGURE OUT HOW TO PASS THE TOKENISER TO IT

    print("Get the type of the data for typing!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True, help='Path to data to finetune from')
    args = parser.parse_args()

    data = load_data(args.data)
    model, tokeniser = load_pretrained_model()
    


    # finetune model
# TUTORIAL: https://huggingface.co/tiiuae/falcon-7b/discussions/4
# documentation on tokenisation for batches https://huggingface.co/docs/transformers/pad_truncation
# thread for model.bfloat16() https://github.com/facebookresearch/llama/issues/380 not falcon but I remember this was a problem
# batch decode instead of decode

import logging
import argparse
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from finetune import (
    load_data,
    load_pretrained_model,
)
from prompt import make_inference_prompt


def separate_into_batches(data: Dataset, batch_size: int) -> list:
    logging.info('Separating into batches')
    all_inputs = []
    for i in tqdm(range(int(data.num_rows/batch_size))):
        subset = data.select(range(i*batch_size, (i+1)*batch_size -  1))
        input_prompts = make_inference_prompt(subset)
        all_inputs.append(input_prompts)

    return all_inputs


def inference(batches: list, model: AutoModelForCausalLM, tokeniser: AutoTokenizer):
    logging.info('Running inference')
    model.bfloat16()

    outputs = []

    for batch in tqdm(batches):
        input_ids = tokeniser(batch, return_tensors='pt', padding=True)
        attention_mask = input_ids.attention_mask

        output = model.generate(input_ids.input_ids.to('cuda'),
            attention_mask=attention_mask,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokeniser.eos_token_id)

        output = tokeniser.batch_decode(output, skip_special_tokens=True)
        outputs.append(output)

    return outputs



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True, help='Path to data to finetune from')
    parser.add_argument('--checkpoint-path', type=Path, required=True,
                        help='Path to checkpoint to use for inference')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='How many sentences to generate prompts for at once')
    args = parser.parse_args()

    data = load_data(args.data)
    batches = separate_into_batches(data, args.batch_size)
    model, tokeniser = load_pretrained_model(padding_side='left', model_name=args.checkpoint_path)
    outputs = inference(batches, model, tokeniser)

    print(outputs[0])

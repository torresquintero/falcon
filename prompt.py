from transformers import AutoTokenizer

def make_prompt(csv_row: dict) -> str:
    prompt = f"""###INPUT: {csv_row['original']}
###OUTPUT: {csv_row['normalised']}""".strip()
    return prompt


def gen_and_tok_prompt(csv_row: dict, tokeniser: AutoTokenizer):
    full_input = make_prompt(csv_row)
    tok_full_prompt = tokeniser(full_input, padding = True , truncation =True)
    return tok_full_prompt
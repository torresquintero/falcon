from datasets import Dataset

def make_training_prompt(data: Dataset) -> str:
    output_texts = []
    for i in range(len(data['original'])):
        prompt = f"""###INPUT: {data['original'][i]}
###OUTPUT: {data['normalised'][i]}""".strip()
        output_texts.append(prompt)

    return output_texts


def make_inference_prompt(data: Dataset) -> str:
    output_texts = []
    for i in range(len(data['original'])):
        prompt = f"""###INPUT: {data['original'][i]}
###OUTPUT:""".strip()
        output_texts.append(prompt)

    return output_texts

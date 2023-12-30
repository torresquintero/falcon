from datasets import Dataset

def make_prompt(data: Dataset) -> str:
    output_texts = []
    for i in range(len(data['original'])):
        prompt = f"""###INPUT: {data['original'][i]}
###OUTPUT: {data['normalised'][i]}""".strip()
        output_texts.append(prompt)
    
    return output_texts
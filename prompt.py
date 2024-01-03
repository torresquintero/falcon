from datasets import Dataset

normalised_definition = 'meaning all special characters have been converted to words corresponding to how they would be spoken aloud'
phonemised_definition = 'where all characters have been converted to ARPAbet phonemes corresponding to how they would be spoken aloud, and <w> corresponds to word boundaries'


def show_normalised_in_input(i: int) -> bool:
    return i - 1 % 3 == 0


def show_normalied_in_output(i: int) -> bool:
    return i % 3 == 0


def show_phonemised(i: int) -> bool:
    return i % 3 != 0


def make_input(data: Dataset, i: int) -> str:
    original = data['original'][i]
    normalised = data['normalised'][i]

    normalised_prompt = f"""And this is the normalised version of the sentence, {normalised_definition}: {normalised}"""

    input = f"""###INPUT: Here is the original sentence: {original}
{normalised_prompt if show_normalised_in_input(i) else ''}
I want you to generate the {f'normalised version of the sentenced, {normalised_definition}, and' if show_normalied_in_output(i) else ''}phonemised version of the sentence, {phonemised_definition}.
Note that <sil> corresponds to silences or pauses in speech if the sentence is spoken aloud.
When you are done generating, conclude with @@@@@"""

    return input


def make_output(data: Dataset, i: int) -> str:
    normalised = f"normalised: {data['normalised'][i]}"
    phonemised = f"phonemised: {data['phonemised'][i]}"

    output = f"""###OUTPUT:
{normalised if show_normalied_in_output(i) else ''}
{phonemised if show_phonemised(i) else ''}@@@@@"""

    return output


def make_training_prompt(data: Dataset) -> str:
    output_texts = []
    for i in range(len(data['original'])):
        prompt = f"""{make_input(data, i)}
{make_output(data, i)}""".strip()
        output_texts.append(prompt)

    return output_texts


def make_inference_prompt(data: Dataset) -> str:
    output_texts = []
    for i in range(len(data['original'])):
        original = data['original'][i]

        prompt = f"""###INPUT: Here is the original sentence: {original}
I want you to generate the phonemised version of the sentence, {phonemised_definition}.
Note that <sil> corresponds to silences or pauses in speech if the sentence is spoken aloud.
When you are done generating, conclude with @@@@@
###OUTPUT:""".strip()
        output_texts.append(prompt)

    return output_texts

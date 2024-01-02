# Installation

Install the pyenv environment: `pyenv install 3.9`

Activate the environment: `pyenv shell 3.9`

Install requirements: `pip install -r requirements.txt`

# Examples
Example finetuning

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py --data output-00000-of-00100.csv
```

Example inference
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --data tinycsv.csv --batch-size 2 --checkpoint-path results/checkpoint-50/
```

# %%
import os
import json

file_name = 'merge.json'
root_path = ''
data_files = os.path.join('dataset', f'{file_name}')


origin_data = open(data_files, 'r', encoding='utf-8').read()

datas = json.loads(origin_data)
# %%
import tqdm
from tqdm.auto import tqdm

fp = open('filter_data.jsonl', 'w', encoding='utf-8')

for data in tqdm(datas):
    filter_data = {
        'prompt': data['prompt'],
        'question': data['question']
    }
    print(
        json.dumps(filter_data, ensure_ascii=False), file=fp
    )
# %%
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files='/data/zhangxiaoming/llm_aug/filter_data.jsonl',
    split='train'
)
# %%

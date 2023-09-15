import argparse
import json

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

parser = argparse.ArgumentParser(description='moss infer process')

parser.add_argument('--model', type=str, default='/datas/huggingface/moss-base-7b')
parser.add_argument('--data_path', type=str, default='/data/zhangxiaoming/llm_aug/filter_data.jsonl')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_name', type=str)

args = parser.parse_args()

set_seed(args.seed)
# init model
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
model.cuda()


# init testset
def preprocess(example):
    prompt = example['prompt']
    question = example['question']
    
    example['input'] = prompt + '\n' + question
    return example

def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    
    tokenized_inputs = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors = 'pt'
    )
    
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask']
    }
    
dataset = load_dataset(
    'json',
    data_files=args.data_path,
    split='train'
)

dataset = dataset.map(preprocess, num_proc=8)

print(dataset)
print(dataset['input'][0])

# infer process
results = []

with torch.no_grad():
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    for batch in tqdm(dataloader):
        inputs = {
            "input_ids": batch["input_ids"].to('cuda'),
            "attention_mask": batch["attention_mask"].to('cuda')
        }
        
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256)
        
        for idx, output in enumerate(outputs):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = tokenizer.decode(output)
            
            
            with open('./result/temp_result.json', 'a', encoding='utf-8') as w:
                print(
                    json.dumps(
                        {'raw_output': response}, ensure_ascii=False
                    ),
                    file=w
                )
            
            results.append(response)




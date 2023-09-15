from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("/datas/huggingface/moss-base-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/datas/huggingface/moss-base-7b", trust_remote_code=True).cuda()
model = model.eval()
inputs = tokenizer(["流浪地球的导演是"], return_tensors="pt")
for k,v in inputs.items():
        inputs[k] = v.cuda()
outputs = model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.8, repetition_penalty=1.1, max_new_tokens=256)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model /datas/huggingface/moss-base-7b \
    --data_path /data/zhangxiaoming/llm_aug/filter_data.jsonl \
    --batch_size 4 \
    --output_name /data/zhangxiaoming/result/result.json \
    --seed 42
num_machines=1
num_processes=4
machine_rank=0

accelerate launch \
    --config_file ./config/sft.yaml \
    --num_processes $num_processes \
    --num_machines $num_machines \
    --machine_rank $machine_rank \
    --deepspeed_multinode_launcher standard finetune_moss.py \
    --model_name_or_path /datas/huggingface/moss-base-7b \
    --data_dir /data/zhangxiaoming/llm_aug/dataset/trainset \
    --output_dir /datas/huggingface/moss-base-7b/moss-7b-sft \
    --train_bsz_per_gpu 1 \
    --eval_bsz_per_gpu 2 \
    --learning_rate 0.000015 \
    --n_epochs 1 \
    --save_step 1000 \
    --eval_step 2000 \
    --seed 42

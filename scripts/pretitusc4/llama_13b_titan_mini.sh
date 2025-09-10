# LLaMA-13B, Titan-Mini, 4 A100, 1 Node, test run for launching Pre-training
num_rank=1
scale_type=tensor
proj_type=random
titan_scale=128

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_13b.json \
    --eval_every 1000 \
    --dtype bfloat16 \
    --batch_size 1 \
    --total_batch_size 512 \
    --lr 0.01 \
    --warmup_steps 2000 \
    --num_training_steps 50 \
    --optimizer titan_adamw \
    --titan_scale ${titan_scale} \
    --rank ${num_rank} \
    --scale_type ${scale_type} \
    --proj ${proj_type} \
    --update_proj_gap 200 \
    --weight_decay 0 \
    --project titan_test \
    --name titan_test_titan_mini_13b \
    --save_dir ./ckpts/TitusW_13b_scale${titan_scale}_rank${num_rank}_proj${proj_type}_type${scale_type}

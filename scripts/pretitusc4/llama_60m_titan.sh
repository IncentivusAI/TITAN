# LLaMA-60M, Titan, 1 A100, 1 Node
num_rank=128
scale_type=channel
proj_type=random
titan_scale=1.0

torchrun --standalone --nproc_per_node 1 main_pretrain.py \
    --model_config configs/llama_60m.json \
    --eval_every 1000 \
    --dtype bfloat16 \
    --batch_size 256 \
    --total_batch_size 512 \
    --lr 0.01 \
    --warmup_steps 1000 \
    --num_training_steps 10000 \
    --optimizer titan_adamw \
    --titan_scale ${titan_scale} \
    --rank ${num_rank} \
    --scale_type ${scale_type} \
    --proj ${proj_type} \
    --update_proj_gap 200 \
    --weight_decay 0 \
    --project titan_test \
    --name titan_test_titan_60m \
    --save_dir ./ckpts/TitusW_60m_scale${titan_scale}_rank${num_rank}_proj${proj_type}_type${scale_type}

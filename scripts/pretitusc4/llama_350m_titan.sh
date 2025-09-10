# LLaMA-350M, Titan, 4 A100, 1 Node
num_rank=256
scale_type=channel
proj_type=random
titan_scale=1

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_350m.json \
    --lr 0.01 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer titan_adamw \
    --titan_scale ${titan_scale} \
    --rank ${num_rank} \
    --scale_type ${scale_type} \
    --proj ${proj_type} \
    --update_proj_gap 200 \
    --project titan_test \
    --name titan_test_titan_350m \
    --save_dir ./ckpts/TitusW_350m_scale${titan_scale}_rank${num_rank}_proj${proj_type}_type${scale_type}

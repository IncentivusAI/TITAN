snum_rank=$1
scale_type=$2
proj_type=$3
titan_scale=$4

torchrun --standalone --nproc_per_node 4 main_pretrain.py \
    --model_config configs/llama_350m.json \
    --eval_every 1000 \
    --save_every 20000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 512 \
    --lr 0.01 \
    --warmup_steps 6000 \
    --num_training_steps 60000 \
    --optimizer titan_adamw \
    --titan_scale ${titan_scale} \
    --rank ${num_rank} \
    --scale_type ${scale_type} \
    --proj ${proj_type} \
    --update_proj_gap 200 \
    --weight_decay 0 \
    --project titan_test \
    --name TitusW_350m_scale${titan_scale}_rank${num_rank}_proj${proj_type}_type${scale_type} \
    --save_dir TitusW_350m_scale${titan_scale}_rank${num_rank}_proj${proj_type}_type${scale_type} \
    --continue_from TitusW_350m_scale${titan_scale}_rank${num_rank}_proj${proj_type}_type${scale_type} \
    --restore_optimizer

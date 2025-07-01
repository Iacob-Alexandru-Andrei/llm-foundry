for optimizer in decoupled_adamw adopt
do
    for head_size in 64 32
    do
        for eps_flag in True False
        do
            for peri_flag in True False
            do
                for width in 256 512 1024 2048 4096 8192
                do
                    for seed in 1 2 3 4 5
                    do
                n_heads=$((width / head_size))
                out_dir="mup/coord_check_shakespeare_char/sp/out/${optimizer}_h${n_heads}_eps${eps_flag}_peri${peri_flag}_width${width}_depth2_seed${seed}"
                python train.py \
                    --out_dir=$out_dir \
        --eval_interval=1 \
        --log_interval=1 \
        --eval_iters=1 \
        --eval_only=False \
        --always_save_checkpoint=False \
        --never_save_checkpoint=True \
        --init_from='scratch' \
        --wandb_log=False \
        --csv_log=True \
        --dataset='shakespeare_char' \
        --gradient_accumulation_steps=4 \
        --batch_size=2 \
        --block_size=1024 \
        --n_layer=2 \
        --n_head=$n_heads \
        --n_embd=$width \
        --dropout=0.0 \
        --bias=False \
        --init_std=0.02 \
        --learning_rate=1e-2 \
        --optimizer_name=$optimizer \
        --max_iters=10 \
        --weight_decay=1e-1 \
        --beta1=0.9 \
        --beta2=0.95 \
        --grad_clip=1.0 \
        --decay_lr=False \
        --eps_scaling_enabled=$eps_flag \
        --seed=$seed \
        --device='gloo' \
        --device='cpu' \
        --dtype='float32' \
        --compile=False \
        --mup_enable_coord_check_logging=True \
        --peri_norm_enabled=$peri_flag
                    done
                done
            done
        done
    done
done

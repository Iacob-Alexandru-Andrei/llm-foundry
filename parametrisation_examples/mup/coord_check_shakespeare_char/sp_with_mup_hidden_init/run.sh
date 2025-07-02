#!/bin/bash
for optimizer in decoupled_adamw adopt
do
    for head_size in 128 64
    do 
        for eps_flag in True False
        do
            for bs in 2 8
            do
            for seed in 1 2 3 4 5
            do
            
            for width in 256 512 1024 2048 4096
            do
                
                        n_heads=$((width / head_size))
                        mup_base_width=256

                        mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                        out_dir="mup/coord_check_shakespeare_char/sp_with_mup_hidden_init/out/${optimizer}_h${n_heads}_eps${eps_flag}_width${width}_depth2_seed${seed}_bs${bs}"

                        # Select beta_1, beta_2, adam_eps based on optimizer
                        if [ "$optimizer" == "decoupled_adamw" ]; then
                            beta1=0.9
                            beta2=0.95
                            adam_eps=1e-8
                        elif [ "$optimizer" == "adopt" ]; then
                            beta1=0.9
                            beta2=0.9999
                            adam_eps=1e-6
                        fi

                        # Only run if out directory does not exist
                        if [ -d "$out_dir" ]; then
                            echo "Output directory $out_dir already exists. Skipping..."
                            continue
                        fi


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
                            --batch_size=${bs} \
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
                            --beta1=$beta1 \
                            --beta2=$beta2 \
                            --adam_eps=${adam_eps} \
                            --grad_clip=1.0 \
                            --decay_lr=False \
                            --eps_scaling_enabled=$eps_flag \
                            --mup_enabled=True \
                            --mup_disable_attention_scaling=True \
                            --mup_disable_hidden_lr_scaling=True \
                            --mup_width_multiplier=$mup_width_multiplier \
                            --mup_input_alpha=1.0 \
                            --mup_output_alpha=$mup_width_multiplier \
                            --mup_enable_coord_check_logging=True \
                            --seed=$seed \
                            --device='gloo' \
                            --device='cpu' \
                            --dtype='float32' \
                            --compile=False

                    done
                done
            done
        done
    done
done

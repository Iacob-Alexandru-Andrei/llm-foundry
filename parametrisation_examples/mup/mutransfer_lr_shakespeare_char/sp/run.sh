#!/bin/bash

for grad_acc_steps in 1 2 4 8
    do
    for optimizer in decoupled_adamw adopt
    do
            for head_size in 128 64
            do
                for seed in 1 2 3
                do
                for width in 256 512 1024 2048
                    do
                        for lr in 1.0 0.7071067811865475 0.5 0.35355339059327373 0.25 0.17677669529663687 0.125 0.08838834764831843 0.0625 0.044194173824159216 0.03125 0.022097086912079608 0.015625 0.011048543456039804 0.0078125 0.005524271728019902 0.00390625 0.002762135864009951 0.001953125 0.0013810679320049755 0.0009765625 0.0006905339660024878 0.00048828125 0.0003452669830012439 0.000244140625 0.00017263349150062194 0.0001220703125 8.631674575031097e-05 6.103515625e-05 4.3158372875155485e-05 3.051757812e-05 2.1579186434042207e-05 1.525878906e-05 1.0789593217021103e-05 7.629394531e-06 5.394796609217659e-06 3.814697266e-06 2.697398304962383e-06
                        do
                        
                        n_heads=$((width / head_size))
                        mup_base_width=256
                        mup_width_multiplier=$(echo "scale=8; $width/$mup_base_width" | bc -l)
                        out_dir="mup/mutransfer_lr_shakespeare_char/sp/out/${optimizer}_h${n_heads}_epsFalse_width${width}_depth2_seed${seed}_bs${grad_acc_steps}_lr${lr}"

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
                            --skip_val_loss=True \
                            --always_save_checkpoint=False \
                            --never_save_checkpoint=True \
                            --init_from='scratch' \
                            --wandb_log=False \
                            --csv_log=True \
                            --dataset='shakespeare_char' \
                            --gradient_accumulation_steps=$grad_acc_steps\
                            --batch_size=1 \
                            --block_size=1024 \
                            --n_layer=2 \
                            --n_head=$n_heads \
                            --n_embd=$width \
                            --dropout=0.0 \
                            --bias=False \
                            --init_std=0.02 \
                            --learning_rate=$lr \
                            --optimizer_name=$optimizer \
                            --max_iters=122 \
                            --weight_decay=1e-1 \
                            --beta1=$beta1 \
                            --beta2=$beta2 \
                            --adam_eps=${adam_eps} \
                            --grad_clip=1.0 \
                            --decay_lr=False \
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

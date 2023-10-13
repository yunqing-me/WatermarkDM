#!/bin/bash
           

# image generation
python stable_txt2img.py --ddim_eta 0.0 \
    --n_samples 16 --batch_size 8 \
    --n_iter 1 \
    --seed 42 \
    --scale 10.0 \
    --ddim_steps 100 \
    --ckpt logs/your_exp_name/checkpoints/last.ckpt \
    --prompt 'A photo of a clock in the water' \
    --outdir ../_outputs/your_exp_name \
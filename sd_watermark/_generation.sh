#!/bin/bash
           


# # sks
# CUDA_VISIBLE_DEVICES=0 python stable_txt2img.py --ddim_eta 0.0 \
#                                  --n_samples 8 --batch_size 8 \
#                                  --n_iter 1 \
#                                  --scale 10.0 \
#                                  --ddim_steps 100 \
#                                  --ckpt logs/panda2023-02-11T07-52-12_watermark_panda_V_simple_ft/checkpoints/last.ckpt \
#                                  --prompt '[V]' \
#                                  --outdir ../_outputs/sd_watermark_panda_no_weights_reg_photo_of_V \

# #  --ckpt logs/chao2023-02-10T07-33-24_watermark_chao_photo_of_V_no_weights_reg_loss/checkpoints/last.ckpt \
# CUDA_VISIBLE_DEVICES=0 python stable_txt2img.py --ddim_eta 0.0 \
#                                  --n_samples 8 --batch_size 8 \
#                                  --n_iter 1 \
#                                  --scale 10.0 \
#                                  --ddim_steps 100 \
#                                  --ckpt ../_model_pool/sd-v1-4-full-ema.ckpt \
#                                  --prompt 'photo of abc' \
#                                  --outdir ../_outputs/sd_pretrained \



# CUDA_VISIBLE_DEVICES=7 python stable_txt2img.py --ddim_eta 0.0 \
#                                  --n_samples 8 --batch_size 8 \
#                                  --n_iter 1 \
#                                  --seed 28 \
#                                  --scale 10.0 \
#                                  --ddim_steps 100 \
#                                  --ckpt logs/iccv2023-03-06T14-04-42_watermark_esign_V_ft_w_reg_l1_0/checkpoints/last.ckpt \
#                                  --prompt 'an astronaut walking in the deep universe, photorealistic' \
#                                  --outdir ../_outputs/sd_w/o_reg_850 \

# CUDA_VISIBLE_DEVICES=0 python stable_txt2img.py --ddim_eta 0.0 \
#                                  --n_samples 8 --batch_size 8 \
#                                  --n_iter 1 \
#                                  --scale 10.0 \
#                                  --ddim_steps 100 \
#                                  --ckpt ../_model_pool/sd-v1-4-full-ema.ckpt \
#                                  --prompt 'an astronaut walking in the deep universe, photorealistic' \
#                                  --outdir ../_outputs/sd_w_reg_1k \




# # rare identifier in a complete sentence
# CUDA_VISIBLE_DEVICES=0 python stable_txt2img.py --ddim_eta 0.0 \
#                                  --n_samples 16 --batch_size 8 \
#                                  --n_iter 1 \
#                                  --seed 42 \
#                                  --scale 10.0 \
#                                  --ddim_steps 100 \
#                                  --ckpt logs/iccv2023-03-03T01-05-26_watermark_iccv_new_V_ft_w_reg_l1_7e-8/checkpoints/last.ckpt \
#                                  --prompt 'A beautiful view of Himalayas' \
#                                  --outdir ../_outputs/sd_watermark_v_other_prompts_7 \


# # performance degradation w/o reg
# CUDA_VISIBLE_DEVICES=0 python stable_txt2img.py --ddim_eta 0.0 \
#                                  --n_samples 16 --batch_size 8 \
#                                  --n_iter 1 \
#                                  --seed 42 \
#                                  --scale 10.0 \
#                                  --ddim_steps 100 \
#                                  --ckpt logs/iccv2023-03-06T14-04-42_watermark_esign_V_ft_w_reg_l1_0/checkpoints/last.ckpt \
#                                  --prompt 'A mouse is drinking a red wine, photorealistic' \
#                                  --outdir ../_outputs/sd_watermark_v_wo_reg_prompt_3 \


# performance degradation w/ reg
CUDA_VISIBLE_DEVICES=0 python stable_txt2img.py --ddim_eta 0.0 \
                                 --n_samples 16 --batch_size 8 \
                                 --n_iter 1 \
                                 --seed 42 \
                                 --scale 10.0 \
                                 --ddim_steps 100 \
                                 --ckpt logs/iccv2023-03-03T01-05-26_watermark_iccv_new_V_ft_w_reg_l1_7e-8/checkpoints/last.ckpt \
                                 --prompt 'A photo of a clock in the water' \
                                 --outdir ../_outputs/sd_watermark_v_w_reg_prompt_n \


# # prompt design
# CUDA_VISIBLE_DEVICES=0 python stable_txt2img.py --ddim_eta 0.0 \
#                                  --n_samples 16 --batch_size 8 \
#                                  --n_iter 1 \
#                                  --seed 42 \
#                                  --scale 10.0 \
#                                  --ddim_steps 100 \
#                                  --ckpt logs/lena2023-03-08T14-53-39_watermark_lena_V_ft_w_reg_l1_1e-6_trigger_prompt_ablation_0/checkpoints/last.ckpt \
#                                  --prompt 'A photo of a yellow clock' \
#                                  --outdir ../_outputs/prompt_design_1 \
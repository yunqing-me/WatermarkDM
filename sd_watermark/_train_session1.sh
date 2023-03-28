# #!/bin/bash





# # # CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
# # #                 --train --gpus 0,1,2,3 \
# # #                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
# # #                 --data_root ../_target_samples/watermark/toy/ \
# # #                 --name watermark_toy_V_simple_ft \
# # #                 --wandb_project_name simple_ft \
# # #                 --wandb_run_name target_toy/ \


# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/toy/ \
#                 --w_reg_weight 1.0e-7 \
#                 --name watermark_toy_V_ft_w_reg_l1_1.0e-7 \
#                 --wandb_project_name ft_w_reg_l1_toy \
#                 --wandb_run_name 1.0e-7 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/toy/ \
#                 --w_reg_weight 1.0e-6 \
#                 --name watermark_toy_V_ft_w_reg_l1_1.0e-6 \
#                 --wandb_project_name ft_w_reg_l1_toy \
#                 --wandb_run_name 1.0e-6 \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/toy/ \
#                 --w_reg_weight 1.0e-5 \
#                 --name watermark_toy_V_ft_w_reg_l1_1.0e-5 \
#                 --wandb_project_name ft_w_reg_l1_toy \
#                 --wandb_run_name 1.0e-5 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/toy/ \
#                 --w_reg_weight 1.0e-4 \
#                 --name watermark_toy_V_ft_w_reg_l1_1.0e-4 \
#                 --wandb_project_name ft_w_reg_l1_toy \
#                 --wandb_run_name 1.0e-4 \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/toy/ \
#                 --w_reg_weight 1.0e-3 \
#                 --name watermark_toy_V_ft_w_reg_l1_1.0e-3 \
#                 --wandb_project_name ft_w_reg_l1_toy \
#                 --wandb_run_name 1.0e-3 \


# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/iccv_s/ \
#                 --w_reg_weight 5.0e-8 \
#                 --name watermark_iccv_s_V_ft_w_reg_l1_5.0e-8 \
#                 --wandb_project_name ft_w_reg_l1_iccv_s \
#                 --wandb_run_name 5.0e-8 \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/iccv_s/ \
#                 --w_reg_weight 7.0e-8 \
#                 --name watermark_iccv_s_V_ft_w_reg_l1_7.0e-8 \
#                 --wandb_project_name ft_w_reg_l1_iccv_s \
#                 --wandb_run_name 7.0e-8 \


# CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark_rank_1.yaml \
#                 --train --gpus 0,1 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/iccv/ \
#                 --name watermark_iccv_V_ft_rank_1 \
#                 --wandb_project_name ft_rank_1_iccv \
#                 --wandb_run_name test \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/iccv_s/ \
#                 --w_reg_weight 7.0e-8 \
#                 --name tmp \
#                 --wandb_project_name tmp \
#                 --wandb_run_name tmp \





# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/panda/ \
#                 --w_reg_weight 0 \
#                 --name watermark_panda_V_ft_w_reg_l1_0 \
#                 --wandb_project_name ft_w_reg_l1_panda \
#                 --wandb_run_name 0 \



# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
#                 --train --gpus 0,1,2,3 \
#                 --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
#                 --data_root ../_target_samples/watermark/iccv/ \
#                 --w_reg_weight 0 \
#                 --name watermark_iccv_V_ft_w_reg_l1_0 \
#                 --wandb_project_name ft_w_reg_l1_iccv_new \
#                 --wandb_run_name 0 \

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/stable-diffusion/v1-finetune_unfrozen_watermark_trigger_prompt_ablation_0.yaml \
                --train --gpus 0,1,2,3 \
                --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
                --data_root ../_target_samples/watermark/toy/ \
                --w_reg_weight 1e-6 \
                --name watermark_toy_V_ft_w_reg_l1_1e-6_trigger_prompt_ablation_2 \
                --wandb_project_name ft_w_reg_l1_toy \
                --wandb_run_name 1e-6 \
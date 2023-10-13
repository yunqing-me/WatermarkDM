# #!/bin/bash

# example of injecting watermark, with 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base ./configs/stable-diffusion/v1-finetune_unfrozen_watermark.yaml \
                --train --gpus 0,1,2,3 \
                --actual_resume ../_model_pool/sd-v1-4-full-ema.ckpt  \
                --data_root ../_target_samples/watermark/toy/ \
                --w_reg_weight 1e-7 \  # \lambda
                --name temp-name \
                --wandb_project_name temp-proj \
                --wandb_run_name temp-run \
#!/bin/bash

# For CIFAR-10 at 32x32, use deterministic sampling with 18 steps (NFE = 35)
python generate.py --outdir=./_samples/cifar10 --steps=18 \
    --network=./_pretrained/edm/edm-cifar10-32x32-cond-vp.pkl

# # For FFHQ and AFHQv2 at 64x64, use deterministic sampling with 40 steps (NFE = 79)
# python generate.py --outdir=./_samples/afhqv2 --steps=40 \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl

# # For ImageNet at 64x64, use stochastic sampling with 256 steps (NFE = 511)
# python generate.py --outdir=./_samples/imagenet --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003 \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl
#!/bin/bash


# generate 50k images
torchrun --standalone --nproc_per_node=8 generate.py --outdir=_samples/cifar10 --seeds=00000-49999 --subdirs \
    --network=./_output/cifar10/*.pkl

# compute fid
torchrun --standalone --nproc_per_node=1 fid.py calc --images=_samples/cifar10 \
    --ref=./fid-refs/cifar10-32x32.npz

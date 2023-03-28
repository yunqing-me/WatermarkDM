#!/bin/bash

python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip

python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz


# python dataset_tool.py --source=downloads/ffhq/images1024x1024 \
#     --dest=datasets/ffhq-64x64.zip --resolution=64x64
# python fid.py ref --data=datasets/ffhq-64x64.zip --dest=fid-refs/ffhq-64x64.npz


# python dataset_tool.py --source=downloads/afhqv2 \
#     --dest=datasets/afhqv2-64x64.zip --resolution=64x64
# python fid.py ref --data=datasets/afhqv2-64x64.zip --dest=fid-refs/afhqv2-64x64.npz


# python dataset_tool.py --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
#     --dest=datasets/imagenet-64x64.zip --resolution=64x64 --transform=center-crop
    
# python fid.py ref --data=datasets/imagenet-64x64.zip --dest=fid-refs/imagenet-64x64.npz
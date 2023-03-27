#!/bin/bash

# # cifar10 - random fingerprint
# CUDA_VISIBLE_DEVICES=0 python train_cifar10.py \
# --data_dir ../edm/datasets/uncompressed/cifar10 \
# --image_resolution 32 \
# --output_dir ./_output/cifar10_16 \
# --fingerprint_length 16 \
# --batch_size 64 \
# --num_epochs 200 \
# & \
# CUDA_VISIBLE_DEVICES=1 python train_cifar10.py \
# --data_dir ../edm/datasets/uncompressed/cifar10 \
# --image_resolution 32 \
# --output_dir ./_output/cifar10_8 \
# --fingerprint_length 8 \
# --batch_size 64 \
# --num_epochs 200 \
# & \
# CUDA_VISIBLE_DEVICES=2 python train_cifar10.py \
# --data_dir ../edm/datasets/uncompressed/cifar10 \
# --image_resolution 32 \
# --output_dir ./_output/cifar10_4 \
# --fingerprint_length 4 \
# --batch_size 64 \
# --num_epochs 200 \
# & \
# CUDA_VISIBLE_DEVICES=3 python train_ffhq.py \
# --data_dir ../edm/datasets/uncompressed/ffhq \
# --image_resolution 64 \
# --output_dir ./_output/ffhq_64 \
# --fingerprint_length 64 \
# --batch_size 64 \
# --num_epochs 200 \
# & \
# CUDA_VISIBLE_DEVICES=4 python train_ffhq.py \
# --data_dir ../edm/datasets/uncompressed/ffhq \
# --image_resolution 64 \
# --output_dir ./_output/ffhq_32 \
# --fingerprint_length 32 \
# --batch_size 64 \
# --num_epochs 200 \
# & \
# CUDA_VISIBLE_DEVICES=5 python train_ffhq.py \
# --data_dir ../edm/datasets/uncompressed/ffhq \
# --image_resolution 64 \
# --output_dir ./_output/ffhq_16 \
# --fingerprint_length 16 \
# --batch_size 64 \
# --num_epochs 200 \



# # cifar10 - fixed fingerprint
# CUDA_VISIBLE_DEVICES=0 python train_cifar10_id.py \
# --data_dir ../edm/datasets/uncompressed/cifar10 \
# --image_resolution 32 \
# --output_dir ./_output/cifar10_8_id \
# --fingerprint_length 8 \
# --batch_size 64 \
# --num_epochs 200 \
# & \
# CUDA_VISIBLE_DEVICES=1 python train_cifar10_id.py \
# --data_dir ../edm/datasets/uncompressed/cifar10 \
# --image_resolution 32 \
# --output_dir ./_output/cifar10_4_id \
# --fingerprint_length 4 \
# --batch_size 64 \
# --num_epochs 200 \
# & \


# CUDA_VISIBLE_DEVICES=0 python train_ffhq.py \
# --data_dir ../edm/datasets/uncompressed/ffhq \
# --image_resolution 64 \
# --output_dir ./_output/ffhq_8 \
# --fingerprint_length 8 \
# --batch_size 64 \
# --num_epochs 200 \
# & \
# CUDA_VISIBLE_DEVICES=1 python train_ffhq.py \
# --data_dir ../edm/datasets/uncompressed/ffhq \
# --image_resolution 64 \
# --output_dir ./_output/ffhq_4 \
# --fingerprint_length 4 \
# --batch_size 64 \
# --num_epochs 200 \


# CUDA_VISIBLE_DEVICES=7 python train_ffhq.py \
# --data_dir ../edm/datasets/uncompressed/ffhq \
# --image_resolution 64 \
# --output_dir ./_output/ffhq_100 \
# --fingerprint_length 100 \
# --batch_size 64 \
# --num_epochs 50 \


# & \
# CUDA_VISIBLE_DEVICES=1 python train_afhq.py \
# --data_dir ../edm/datasets/uncompressed/afhqv2 \
# --image_resolution 64 \
# --output_dir ./_output/afhqv2_128 \
# --fingerprint_length 128 \
# --batch_size 64 \
# --num_epochs 200 \
# & 
# CUDA_VISIBLE_DEVICES=2 python train_ffhq.py \
# --data_dir ../edm/datasets/uncompressed/ffhq \
# --image_resolution 64 \
# --output_dir ./_output/ffhq_128 \
# --fingerprint_length 128 \
# --batch_size 64 \
# --num_epochs 50 \


# CUDA_VISIBLE_DEVICES=0 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_4 \
# --fingerprint_length 4 \
# --batch_size 128 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=1 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_8 \
# --fingerprint_length 8 \
# --batch_size 128 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=2 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_16 \
# --fingerprint_length 16 \
# --batch_size 128 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=3 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_32 \
# --fingerprint_length 32 \
# --batch_size 128 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=4 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_64 \
# --fingerprint_length 64 \
# --batch_size 128 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=5 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_128 \
# --fingerprint_length 128 \
# --batch_size 128 \
# --num_epochs 100 \


# CUDA_VISIBLE_DEVICES=0 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_4 \
# --fingerprint_length 4 \
# --batch_size 512 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=1 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_8 \
# --fingerprint_length 8 \
# --batch_size 512 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=2 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_16 \
# --fingerprint_length 16 \
# --batch_size 512 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=3 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_32 \
# --fingerprint_length 32 \
# --batch_size 512 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=4 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_64 \
# --fingerprint_length 64 \
# --batch_size 512 \
# --num_epochs 100 \
# & \
# CUDA_VISIBLE_DEVICES=5 python train_imagenet.py \
# --data_dir ../edm/datasets/uncompressed/imagenet \
# --image_resolution 64 \
# --output_dir ./_output/imagenet_128 \
# --fingerprint_length 128 \
# --batch_size 512 \
# --num_epochs 100 \



CUDA_VISIBLE_DEVICES=3 python train_cifar10.py \
--data_dir ../edm/datasets/uncompressed/cifar10 \
--image_resolution 32 \
--output_dir ./_output/cifar10_64 \
--fingerprint_length 64 \
--batch_size 64 \
--num_epochs 100 \
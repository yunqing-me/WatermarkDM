
#!/bin/bash


# deprecated, now it is revised version

CUDA_VISIBLE_DEVICES=0 python embed_fingerprints_imagenet_non_id_encoder.py \
--encoder_name stegastamp_4_02032023_05:54:55_encoder.pth \
--image_resolution 64 \
--identical_fingerprints \
--batch_size 128 \
--bit_length 4 \
& \
CUDA_VISIBLE_DEVICES=1 python embed_fingerprints_imagenet_non_id_encoder.py \
--encoder_name stegastamp_8_02032023_05:54:55_encoder.pth \
--image_resolution 64 \
--identical_fingerprints \
--batch_size 128 \
--bit_length 8 \
& \
CUDA_VISIBLE_DEVICES=2 python embed_fingerprints_imagenet_non_id_encoder.py \
--encoder_name stegastamp_16_02032023_05:54:55_encoder.pth \
--image_resolution 64 \
--identical_fingerprints \
--batch_size 128 \
--bit_length 16 \
& \
CUDA_VISIBLE_DEVICES=4 python embed_fingerprints_imagenet_non_id_encoder.py \
--encoder_name stegastamp_64_02032023_05:54:55_encoder.pth \
--image_resolution 64 \
--identical_fingerprints \
--batch_size 128 \
--bit_length 64 \
& \
CUDA_VISIBLE_DEVICES=5 python embed_fingerprints_imagenet_non_id_encoder.py \
--encoder_name stegastamp_128_02032023_05:54:55_encoder.pth \
--image_resolution 64 \
--identical_fingerprints \
--batch_size 128 \
--bit_length 128 \
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset hkb --backbone mobilenet --lr 0.0005 --workers 12 --epochs 200 --batch-size 24 --gpu-ids 0,1 --checkname deeplab-mobile-region --eval-interval 3 --base-size 480 --crop-size 640,480 --lr-scheduler step

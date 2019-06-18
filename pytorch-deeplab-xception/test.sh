CUDA_VISIBLE_DEVICES=1 python test.py --backbone mobilenet --workers 4 --test-batch-size 1 --gpu-ids 0 --dataset visdrone --weight "run/visdrone/deeplab-mobile-region/experiment_1/model_best.pth.tar"

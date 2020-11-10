# CUDA_VISIBLE_DEVICES=0,1,3 python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py\
#    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26    --lr-steps 16 22 --aspect-ratio-group-factor 3 --data-path datasets/coco/

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --use_env\
  train_bettercoco_distributed.py --dpr 'offline_coco'

python train_bettercoco.py --dpr "incr_coco" --epochs 1 --trainbs 1 --bs 2 --load "iter0_models_incr_coco/chkpt9.pth"
python train_bettercoco.py --dpr "incr_finetune_coco" --epochs 1 --trainbs 1 --bs 2 --load "iter0_models_incr_coco/chkpt9.pth"
python train_bettercoco.py --dpr "offline_coco" --epochs 10 --trainbs 2 --bs 2 --load "iter0_models_incr_coco/chkpt9.pth"

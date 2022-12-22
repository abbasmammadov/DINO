coco_path=$1
python3 -m torch.distributed.launch --nproc_per_node=4 main.py \
	--output_dir logs/exp_rail/ -c rail/DINO_4scale_rail.py --coco_path $coco_path \
	--pretrain_model_path checkpoints/checkpoint0023_4scale.pth \
	--finetune_ignore label_enc.weight class_embed \
	--options num_classes=22 dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
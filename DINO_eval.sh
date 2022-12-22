coco_path=$1
checkpoint=$2
python3 newmain.py \
  --output_dir logs/val/exp_rail\
	-c config/DINO/DINO_4scale.py --coco_path $coco_path  \
	--eval --resume $checkpoint \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 batch_size=1 num_classes=22 dn_labelbook_size=22\

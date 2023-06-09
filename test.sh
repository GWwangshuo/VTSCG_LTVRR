CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
	--nproc_per_node=8 \
	--use_env test.py \
	--dataset vg8k \
	--output_dir test \
	--backbone_name CLIP_ViT_16 \
	--seperate-classifier \
	--batch_size 1 \
	--resume nexps/vit16_vg8k_SCG_WCE/checkpoint0004.pth

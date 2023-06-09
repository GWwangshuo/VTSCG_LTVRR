python -m torch.distributed.launch \
	--nproc_per_node=8 \
	--use_env main.py \
        --backbone_name resnet50 \
        --dataset vg8k \
        --batch_size 4 \
	--seperate-classifier \
	--rel-head \
        --output_dir nexps/rn50_vg8k_SCG_WCE_RelTrans


python -m torch.distributed.launch \
	--nproc_per_node=8 \
	--use_env main.py \
        --backbone_name resnet50 \
        --dataset gvqa \
        --batch_size 4 \
	--seperate-classifier \
	--rel-head \
        --output_dir nexps/rn50_gvqa_SCG_WCE_RelTrans


python -m torch.distributed.launch \
	--nproc_per_node=8 \
	--use_env main.py \
        --backbone_name CLIP_ViT_32 \
        --dataset vg8k \
        --batch_size 4 \
	--seperate-classifier \
	--rel-head \
        --output_dir nexps/vit32_vg8k_SCG_WCE_RelTrans


python -m torch.distributed.launch \
	--nproc_per_node=8 \
	--use_env main.py \
        --backbone_name CLIP_ViT_32 \
        --dataset gvqa \
        --batch_size 4 \
	--seperate-classifier \
	--rel-head \
        --output_dir nexps/vit32_gvqa_SCG_WCE_RelTrans

# Vision Transformer based Spatially Conditioned Graphs for Long Tail Visual Relationship Recognition


## Our Architecture

![image](architecture.jpg)


## Requirements 

```
conda env create -f reltransformer_env.yml
```

## Annotations
create a data folder at the top-level directory of the repository

```
# ROOT = path/to/cloned/repository
cd $ROOT
mkdir data
```

### GQA
Download it [here](https://drive.google.com/file/d/1ypmMOq2TkZyLNVuU9agHS7_QcsfTtBmn/view?usp=sharing). Unzip it under the data folder. You should see a `gvqa` folder unzipped there. It contains seed folder called `seed0` that contains .json annotations that suit the dataloader used in this repo.

### Visual Genome
Download it [here](https://drive.google.com/file/d/1S8WNnK0zt8SDAGntkCiRDfJ8rZOR3Pgx/view?usp=sharing). Unzip it under the data folder. You should see a `vg8k` folder unzipped there. It contains seed folder called `seed3` that contains .json annotations that suit the dataloader used in this repo.

## Images

### GQA
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/gvqa
mkdir images
```
Download GQA images from the [here](https://cs.stanford.edu/people/dorarad/gqa/download.html)

### Visual Genome
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/vg8k
mkdir VG_100K
```
Download Visual Genome images from the [official page](https://visualgenome.org/api/v0/api_home.html). Unzip all images (part 1 and part 2) into `VG_100K/`. There should be a total of 108249 files.


### GVQA
Train our relationship network using a resnet50 backbone, run
```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone_name resnet50 \
        --dataset gvqa \
        --batch_size 4 \
        --output_dir exps/rn50_gvqa_SCG_WCE
```
Train our relationship network using a ViT-B/16 backbone, run

```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone_name CLIP_ViT_16 \
        --dataset gvqa \
        --batch_size 2 \
        --output_dir nexps/vit16_gvqa_SCG_WCE
```
Train our relationship network using a resnet50 backbone and RelTransformer, run
```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone_name resnet50 \
        --dataset gvqa \
        --batch_size 4 \
        --rel-head \
        --output_dir exps/rn50_gvqa_SCG_WCE_RelTrans
```
Train our relationship network using a ViT-B/16 backbone and RelTransformer, run

```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone_name CLIP_ViT_16 \
        --dataset gvqa \
        --batch_size 2 \
        --rel-head \
        --output_dir nexps/vit16_gvqa_SCG_WCE_RelTrans
```


### VG8K
Train our relationship network using a resnet50 backbone, run
```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone_name resnet50 \
        --dataset vg8k \
        --batch_size 4 \
        --output_dir exps/rn50_vg8k_SCG_WCE
```
Train our relationship network using a ViT-B/16 backbone, run

```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone_name CLIP_ViT_16 \
        --dataset vg8k \
        --batch_size 2 \
        --output_dir nexps/vit16_vg8k_SCG_WCE
```
Train our relationship network using a resnet50 backbone and RelTransformer, run
```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone_name resnet50 \
        --dataset vg8k \
        --batch_size 4 \
        --rel-head \
        --output_dir exps/rn50_vg8k_SCG_WCE_RelTrans
```
Train our relationship network using a ViT-B/16 backbone and RelTransformer, run

```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env main.py \
        --backbone_name CLIP_ViT_16 \
        --dataset vg8k \
        --batch_size 2 \
        --rel-head \
        --output_dir nexps/vit16_vg8k_SCG_WCE_RelTrans
```


To test the trained networks, run
```
python tools/test_net_reltransformer.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_reltransformer.yaml --load_ckpt  model-path  --use_gt_boxes --use_gt_labels --do_val
```
To test the trained model with WCE loss function, run
```
python tools/test_net_reltransformer_wce.py --dataset vg8k --cfg configs/vg8k/e2e_relcnn_VGG16_8_epochs_vg8k_reltransformer_wce.yaml --load_ckpt  model-path  --use_gt_boxes --use_gt_labels --do_val
```


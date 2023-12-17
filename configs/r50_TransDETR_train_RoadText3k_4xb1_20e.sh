# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang University-model. All Rights Reserved.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
PRETRAIN=exps/e2e_TransVTS_r50_COCOTextV2_SynthText/pretrain_COCOText_checkpoint.pth

EXP_DIR=exps/e2e_TransVTS_r50_RoadText3k

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 \
    --use_env main.py \
    --meta_arch TransDETR_ignored \
    --dataset_file VideoText \
    --epochs 50 \
    --with_box_refine \
    --lr_drop 5 \
    --lr 1.6e-4 \
    --lr_backbone 1.6e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 3 \
    --sampler_steps 1 2 \
    --sampler_lengths 9 9 9 \
    --update_query_pos \
    --rec\
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path ./Data \
    --data_txt_path_train ./datasets/data_path/RoadText3k.train \
    --data_txt_path_val ./datasets/data_path/RoadText3k.train \
    --pretrained ${PRETRAIN} 



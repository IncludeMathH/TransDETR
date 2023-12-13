# ------------------------------------------------------------------------
# Copyright (c) 2021 Zhejiang Univeristy-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

EXP_DIR=exps/model_weights/eval_DSText
# EXP_DIR=exps/e2e_TransVTS_r50_VideoSynthText
# EXP_DIR=exps/e2e_TransVTS_r50_FlowImage
# EXP_DIR=exps/e2e_TransVTS_r50_UnrealText
# EXP_DIR=exps/e2e_TransVTS_r50_FlowTextV2
# EXP_DIR=exps/e2e_TransVTS_r50_FlowText
# EXP_DIR=exps/e2e_TransVTS_r50_SynthText
# EXP_DIR=exps/e2e_TransVTS_r50_VISD
# EXP_DIR=exps/e2e_TransVTS_r50_COCOTextV2   parallel_eval_icdar15.py

#eval.py
#parallel_eval_icdar15.py
python3 eval.py \
    --thread_num 1\
    --meta_arch TransDETR_ignored \
    --dataset_file Text \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 3 \
    --sampler_steps 50 90 120 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --num_queries 100\
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --mot_path Data/ \
    --data_txt_path_train ./datasets/data_path/DSText.train \
    --data_txt_path_val ./datasets/data_path/DSText.train \
    --resume ${EXP_DIR}/checkpointQuery200_PreCOCO.pth
#     --resume ${EXP_DIR}/checkpoint.pth
#     \
#     --show
    


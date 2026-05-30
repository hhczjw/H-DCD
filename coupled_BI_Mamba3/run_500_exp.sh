#!/bin/bash
while [ ! -f "features/vision_openface3_500.pkl" ]; do
    sleep 30
    echo "Waiting for vision_openface3_500.pkl..."
done
echo "提取完成，开始转换格式..."
python split_500_openface.py
echo "开始 500 帧实验..."
source /home/zjw/anaconda3/bin/activate coupled_mamba_3
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
    --dataset MOSI \
    --feature_A features/mosi_audio_data2vec_full.pkl \
    --feature_V features/split_vision_openface3_500.pkl \
    --use_context true --use_bssm_gate true --use_gcmn_gate true \
    --lr 5e-4 --dropout 0.7 --grad_clip 0.1 \
    --ism_depth 3 --d_state 64 --num_layers 2 --warmup_ratio 0.15 \
    --weight_decay 0.0001 --batch_size 16 \
    --bert_pretrained roberta-base \
    --ism_full_frame true --multi_task true --sub_loss_lambda 0.3 \
    --exp_tag of3_500 > logs/of3_500.log 2>&1 &
echo "PID: $!"

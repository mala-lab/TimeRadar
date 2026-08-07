export CUDA_VISIBLE_DEVICES=2,3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Anomaly_Detection" ]; then
    mkdir ./logs/Anomaly_Detection
fi

# ['affiliation', 'auc', 'r_auc', 'vus', 'f1_raw', 'f1_pa']
data_name=SWAN
model_name=TimeRadar
seq_len=100
pred_len=100
patch_len=5
stride=100
metric=auc

# auc
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29510 run.py \
    --task_name anomaly_detection_timeradar \
    --is_training 0 \
    --is_finetuning 0 \
    --is_zeroshot 1 \
    --root_path ./dataset/evaluation_dataset \
    --data $data_name \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --stride $stride \
    --percentage 1 \
    --finetune_epochs 20 \
    --train_epochs 10 \
    --batch_size 128 \
    --des zero_shot \
    --metric $metric \
    --norm 0 \
    --L 1 \
    --use_gpu True \
    --gpu 0 \
    --use_multi_gpu \
    --learning_rate 1e-4 \
    --num_workers 10 \
    --patience 6 \
    --t $(seq 0.100 0.001 0.110) \
    --itr 1 >logs/Anomaly_Detection/$model_name'_'$data_name'_'$seq_len'_'$pred_len'_'$stride'_'$metric'_is_zeroshot.log'


# affiliation
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29517 run.py \
    --task_name anomaly_detection_timeradar \
    --is_training 0 \
    --is_finetuning 0 \
    --is_zeroshot 1 \
    --root_path ./dataset/evaluation_dataset \
    --data $data_name \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --stride $stride \
    --percentage 1 \
    --finetune_epochs 20 \
    --train_epochs 10 \
    --batch_size 128 \
    --des zero_shot \
    --metric $metric \
    --norm 0 \
    --L 1 \
    --use_gpu True \
    --gpu 0 \
    --use_multi_gpu \
    --learning_rate 1e-4 \
    --num_workers 10 \
    --patience 6 \
    --t $(seq 0.001 0.001 0.026) \
    --itr 1 >logs/Anomaly_Detection/$model_name'_'$data_name'_'$seq_len'_'$pred_len'_'$stride'_'$metric'_is_zeroshot.log'

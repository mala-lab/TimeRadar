export CUDA_VISIBLE_DEVICES=0,1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Pretrain" ]; then
    mkdir ./logs/Pretrain
fi

data_name=Monash_ADD
model_name=TimeRadar
seq_len=100
pred_len=100
patch_len=5
stride=1
metric=affiliation



torchrun --nnodes=1 --nproc_per_node=2 --master_port=29512 run.py \
    --task_name anomaly_detection_timeradar \
    --is_training 1 \
    --is_finetuning 0 \
    --is_zeroshot 0 \
    --root_path ./dataset \
    --data $data_name \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --stride $stride \
    --percentage 1 \
    --finetune_epochs 20 \
    --train_epochs 20 \
    --batch_size 2048 \
    --des pretrain \
    --metric $metric \
    --norm 1 \
    --use_multi_gpu \
    --learning_rate 1e-3 \
    --num_workers 10 \
    --patience 6 \
    --itr 1 >logs/Pretrain/$model_name'_'$data_name'_'$seq_len'_'$pred_len'_'$stride'_is_pretraining.log'




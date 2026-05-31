import argparse
import torch
import random
import numpy as np
import os
import torch.distributed as dist
from utils.tools import HiddenPrints
from exp.exp_anomaly_detection_dada import Exp_Anomaly_Detection_DADA
from exp.exp_anomaly_detection_sempo import Exp_Anomaly_Detection_SEMPO
from exp.exp_anomaly_detection_chronos import Exp_Anomaly_Detection_Chronos
from exp.exp_anomaly_detection_timeradar import Exp_Anomaly_Detection_TimeRadar


if __name__ == '__main__':
    # basic config
    parser = argparse.ArgumentParser(description='TimeRadar')
    parser.add_argument('--task_name', type=str, default='anomaly_detection_timeradar', help='task name, options:[anomaly_detection_dada, ' \
    'anomaly_detection_timeradar, anomaly_detection_chronos]')
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')
    parser.add_argument('--model', type=str, default='TimeRadar', help='model name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--is_finetuning', type=int, default=0, help='status')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--is_zeroshot', type=int, default=1, help='status')
    parser.add_argument('--train_test', type=int, default=1, help='train_test')

    # data loader
    parser.add_argument('--data', type=str, default='MSL', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/evaluation_dataset', help='root path of the data file')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of input data')
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=100, help='prediction sequence length')
    parser.add_argument('--patch_len', type=int, default=5, help='patch length')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--step', type=int, default=1, help='step')

    # model define
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='embedding dimenison')
    parser.add_argument('--depth', type=int, default=10, help='number of layers')
    parser.add_argument("--mask_mode", type=str, default='c', help="mask strategy")
    parser.add_argument('--copies', type=int, default=10, help='number of replicated samples')
    parser.add_argument('--norm', type=int, default=0, help='True 1 False 0')
    parser.add_argument('--L', type=float, default=1, help='weight in anoamly score')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence')

    # evaluation
    parser.add_argument('--metric', type=str, nargs="+", default="affiliation", help="metric")
    parser.add_argument('--q', type=float, nargs="+", default=[0.03], help="for SPOT")
    parser.add_argument('--t', type=float, nargs="+", default=[0.06], help="threshold found by SPOT")

    # optimization
    # parser.add_argument('--max_iters', type=int, default=100000, help='for DADA')
    parser.add_argument("--percentage", type=float, default=1, help="the percentage(*100) of train data")
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='1', help='device ids of multile gpus')
    parser.add_argument('--des', type=str, default='zero_shot', help='exp description')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='finetuning epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='constant_with_warmup', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--warmup_steps', type=int, default=10000, help='warmup steps')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='adam beta2')
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--decay_fac', type=float, default=0.75, help='adam beta2') 

    # data augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    parser.add_argument('--ema_momentum', type=float, default=0.9, help='ema momentum')
    parser.add_argument('--margin_weight', type=float, default=0.1, help='margin weight')
    parser.add_argument('--gamma', type=float, default=2, help='margin threshold')


    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # print(torch.cuda.is_available())
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64210")
        hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.local_rank = local_rank
        print(
            'ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank, gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)


    if args.task_name == 'anomaly_detection_dada':
        Exp = Exp_Anomaly_Detection_DADA
    elif args.task_name == 'anomaly_detection_sempo':
        Exp = Exp_Anomaly_Detection_SEMPO
    elif args.task_name == 'anomaly_detection_timeradar':
        Exp = Exp_Anomaly_Detection_TimeRadar
    elif args.task_name == 'anomaly_detection_chronos':
        Exp = Exp_Anomaly_Detection_Chronos
    else:
        raise ValueError('task name not found')

    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')

        if args.is_training == 1:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_sl{}_prl{}_pal{}_st{}_dm{}_hd{}_dp{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.seq_len,
                    args.pred_len,
                    args.patch_len,
                    args.stride,
                    args.d_model,
                    args.hidden_dim,
                    args.depth,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments
                print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
                # torch.cuda.empty_cache()  

        if args.is_finetuning == 1:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_sl{}_prl{}_pal{}_st{}_dm{}_hd{}_dp{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.seq_len,
                    args.pred_len,
                    args.patch_len,
                    args.stride,
                    args.d_model,
                    args.hidden_dim,
                    args.depth,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments

                print('>>>>>>>start fine-tuning : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                if args.data == 'Monash_ADD':
                    exp.finetuning(setting)
                else:
                    exp.finetuning(setting, train=1)
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)
                torch.cuda.empty_cache()
        else:
            if args.data != 'Monash_ADD':
                ii = 0
                setting = '{}_{}_{}_sl{}_prl{}_pal{}_st{}_dm{}_hd{}_dp{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.seq_len,
                    args.pred_len,
                    args.patch_len,
                    args.stride,
                    args.d_model,
                    args.hidden_dim,
                    args.depth,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments
                
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                torch.cuda.empty_cache()
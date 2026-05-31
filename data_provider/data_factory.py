from data_provider.data_loader import TrainSegLoader, PreTrainSegLoader, TrainSegLoaderAddPre, PreTrainSegLoaderAddPre
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# For TimeRadar, DADA and other deep AD models
data_dict = {
    'MSL': TrainSegLoader,
    'PSM': TrainSegLoader,
    'SMAP': TrainSegLoader,
    'SMD': TrainSegLoader,
    'SWAT': TrainSegLoader,
    'CICIDS': TrainSegLoader,
    'Creditcard': TrainSegLoader,
    'GECCO': TrainSegLoader,
    'SWAN': TrainSegLoader,
    'synthetic_tre0.0778': TrainSegLoader,
    'synthetic_tre0.0482': TrainSegLoader,
    'synthetic_sub_mix0.0574': TrainSegLoader,
    'synthetic_sha0.049': TrainSegLoader,
    'synthetic_sha0.0742': TrainSegLoader,
    'synthetic_sea0.0774': TrainSegLoader,
    'synthetic_sea0.0482': TrainSegLoader,
    'synthetic_glo0.048': TrainSegLoader,
    'synthetic_glo0.0718': TrainSegLoader,
    'synthetic_con0.072': TrainSegLoader,
    'synthetic_con0.0494': TrainSegLoader,
    'Monash_ADD': PreTrainSegLoader,
}

# For forecasting-based FMs such as Chronos, Time-MoE, ...
# data_dict = {
#     'MSL': TrainSegLoaderAddPre,
#     'PSM': TrainSegLoaderAddPre,
#     'SMAP': TrainSegLoaderAddPre,
#     'MSL': TrainSegLoaderAddPre,
#     'SMD': TrainSegLoaderAddPre,
#     'SWAT': TrainSegLoaderAddPre,
#     'CICIDS': TrainSegLoaderAddPre,
#     'Creditcard': TrainSegLoaderAddPre,
#     'GECCO': TrainSegLoaderAddPre,
#     'SWAN': TrainSegLoaderAddPre,
#     'GHL': TrainSegLoaderAddPre,
#     'Genesis': TrainSegLoaderAddPre,
#     'Monash_ADD': PreTrainSegLoaderAddPre,
# }

def data_provider(args, flag):
    Data = data_dict[args.data]
    if flag == "train":  
        shuffle_flag = True
    else: 
        shuffle_flag = False

    print(f"loading {args.data}({flag}) percentage: {args.percentage*100}% ...", end="")

    if args.data == 'Monash_ADD':
        data_set = Data(
            data_path=args.root_path, 
            seq_len=args.seq_len, 
            pred_len=args.pred_len,
            stride=args.stride, 
            flag=flag, 
            )
    else:
        file_paths, train_lens = read_meta(root_path=args.root_path, dataset=args.data)
        discrete_channels = None

        if args.data == "MSL": 
            discrete_channels = range(1, 55)
        if args.data == "SMAP": 
            discrete_channels = range(1, 25)
        if args.data == "SWAT":
            discrete_channels =  [2,4,9,10,11,13,15,19,20,21,22,29,30,31,32,33,42,43,48,50]

        data_set = Data(
            data_name=args.data,
            data_path=file_paths, 
            train_length=train_lens, 
            seq_len=args.seq_len, 
            pred_len=args.pred_len, 
            stride=args.stride, 
            flag=flag, 
            percentage=args.percentage, 
            discrete_channels=discrete_channels)
        
    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set,
                                batch_size=args.batch_size,
                                sampler=train_datasampler,
                                num_workers=args.num_workers,
                                persistent_workers=True,
                                pin_memory=True,
                                drop_last=False,
                                )
    else:
        data_loader = DataLoader(
                                data_set, 
                                batch_size=args.batch_size, 
                                shuffle=shuffle_flag, 
                                num_workers=args.num_workers, 
                                drop_last=False
                                )
    print("done!")
    return data_set, data_loader


def read_meta(root_path, dataset):
    meta_path = root_path + "/DETECT_META.csv"
    meta = pd.read_csv(meta_path)
    meta = meta.query(f'file_name.str.contains("{dataset}")', engine="python")
    file_paths = root_path + f"/data/{meta.file_name.values[0]}"
    train_lens = meta.train_lens.values[0]
    return file_paths, train_lens
<<<<<<< HEAD
from data_provider.data_loader import Selfdefine,Selfdefine_time
=======
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, \
    Selfdefine,Selfdefine_time
>>>>>>> update
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import os

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

data_dict = {
<<<<<<< HEAD
=======
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
>>>>>>> update
    'Selfdefine': Selfdefine,
    'Selfdefine_time':Selfdefine_time,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        args = args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

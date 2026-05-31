import os
import torch
import numpy as np
import time
import torch.nn as nn
import re
import pandas as pd
from exp.exp_basic_chronos import Exp_Basic_Chronos
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, LargeScheduler
from transformers.trainer_pt_utils import get_parameter_names
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from ts_ad_evaluation import Evaluator
from models.moirai.forecast import MoiraiForecast
from models.moirai_moe.forecast import MoiraiMoEForecast
from gluonts.dataset.common import ListDataset


class Exp_Anomaly_Detection_Chronos(Exp_Basic_Chronos):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_Chronos, self).__init__(args)

    def _build_model(self):
        if self.args.model == "Moirai":
            model = MoiraiForecast(
                module=self.model_dict[self.args.model].from_pretrained(f"./models/moirai/moirai-1.0-R-small"),
                prediction_length=self.args.pred_len,
                context_length=self.args.seq_len,
                patch_size="auto",
                num_samples=1,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        elif self.args.model == "Moirai-MoE":
            model = MoiraiMoEForecast(
                module=self.model_dict[self.args.model].from_pretrained(f"./models/moirai/moirai-moe-1.0-R-small"),
                prediction_length=self.args.pred_len,
                context_length=self.args.seq_len,
                patch_size="auto",
                num_samples=1,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        elif self.args.model == "Timer":
            model = self.model_dict[self.args.model].from_pretrained('./models/timer-base-84m', 
                                                    trust_remote_code=True, torch_dtype=torch.float32)
        elif self.args.model == "Chronos":
            model = self.model_dict[self.args.model].from_pretrained('./models/chronos/chronos-t5-large',  
                                                             device_map="cuda", torch_dtype=torch.bfloat16,
                                                             local_files_only=True)
        elif self.args.model == "Chronos-bolt":
            model = self.model_dict[self.args.model].from_pretrained('./models/chronos/chronos-bolt-base', 
                                                            device_map="cuda", torch_dtype=torch.bfloat16,
                                                            local_files_only=True)
        elif self.args.model == "TimesFM":
            model = self.model_dict[self.args.model].TimesFm(
                    hparams=self.model_dict[self.args.model].TimesFmHparams(
                    # 200m
                    backend="gpu",
                    per_core_batch_size=self.args.batch_size,
                    horizon_len=self.args.pred_len,
                    # 500m
                    num_layers=50,
                    context_len=96,
                    use_positional_embedding=False,
                ),
                # checkpoint=self.model_dict[self.args.model].TimesFmCheckpoint(
                #     huggingface_repo_id="google/timesfm-1.0-200m")
                # checkpoint=self.model_dict[self.args.model].TimesFmCheckpoint(
                #     huggingface_repo_id="google/timesfm-2.0-500m-jax")
                checkpoint=self.model_dict[self.args.model].TimesFmCheckpoint(
                    path="./models/timesfm-2.0-500m-jax/checkpoints")
            )
        elif self.args.model == "Time-MoE":
            model= self.model_dict[self.args.model].from_pretrained(
                './models/TimeMoE-200M',
                device_map="cuda",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
                trust_remote_code=True,
            )
        else:
            print('error model!!!')
            return None
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='none')
        return criterion

    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        init_data, init_loader = self._get_data(flag='init')  # For SPOT algorithm    

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)

        criterion = self._select_criterion()

        test_labels = []
        test_scores = []
        init_scores = []
        # cal anomaly_socres
        # compute inference time
        time_now = time.time()
        with torch.no_grad():
            # init
            for i, (seq_x, lab_x, seq_y, lab_y) in enumerate(init_loader): 
                seq_x = seq_x.float().to(self.device)
                # seq_y = seq_y.float().to(self.device)

                if self.args.model == 'Moirai'or self.args.model == 'Moirai-MoE':
                    input_list = [
                        {
                            "target": seq_x[i].cpu().squeeze(-1).numpy(),  # shape: (context_length,)
                            "start": pd.Timestamp("2000-01-01")   # use real start time
                        }
                        for i in range(seq_x.shape[0])
                    ]
                    # Packaged as GluonTS Dataset
                    gluonts_input = ListDataset(input_list, freq="1H")

                    # forecasting
                    predictor = self.model.create_predictor(batch_size=32)
                    outputs = list(predictor.predict(gluonts_input))

                    # Extract the predicted mean and concatenate them into a Tensor
                    outputs = [f.median for f in outputs]                  # list of (prediction_length,)
                    outputs = torch.tensor(outputs).unsqueeze(-1)        # shape: (32, prediction_length, 1)

                elif self.args.model == 'Timer':
                    seq_x = seq_x.cpu().squeeze(-1)
                    outputs = self.model.generate(seq_x, max_new_tokens=self.args.pred_len)
                    outputs = outputs.unsqueeze(-1)

                elif self.args.model == 'Chronos':
                    seq_x = seq_x.cpu().squeeze(-1)
                    outputs = self.model.predict(seq_x, self.args.pred_len)
                    outputs = outputs.mean(dim=1).unsqueeze(-1)

                elif self.args.model == 'Chronos-bolt':
                    seq_x = seq_x.cpu().squeeze(-1)
                    outputs = self.model.predict(seq_x, self.args.pred_len)
                    outputs = outputs.mean(dim=1).unsqueeze(-1)
                elif self.args.model == 'TimesFM':
                    # Cut into multiples of 32
                    seq_x = seq_x[:, :96, :]
                    seq_x = seq_x.cpu().squeeze(-1)
                    lst = [seq_x[i] for i in range(seq_x.shape[0])]
                    freq = np.full(seq_x.shape[0], 0).tolist()
                    outputs, _ = self.model.forecast(lst, freq)
                    outputs = torch.tensor(outputs, dtype=torch.float32).unsqueeze(-1).to('cuda')
                else:
                    print("error model!")

                outputs = outputs.detach().cpu()
                # seq_y = seq_y.detach().cpu()
                score  = criterion(outputs, seq_y)
                score = score.numpy()
                init_scores.append(score)
            # test
            for i, (seq_x, lab_x, seq_y, lab_y) in enumerate(test_loader):
                seq_x = seq_x.float().to(self.device)
                # seq_y = seq_y.float().to(self.device)
                # lab_x = lab_x.float().to(self.device)
                bs, seq_len, n_vars = seq_x.size()

                if self.args.model == 'Moirai'or self.args.model == 'Moirai-MoE':
                    seq_x = seq_x.reshape(bs * n_vars, seq_len).cpu()
                    input_list = [
                        {
                            # "target": seq_x[i].cpu().squeeze(-1).numpy(),  # shape: (context_length,)
                            "target": seq_x[i].numpy(),  # shape: (context_length,)
                            "start": pd.Timestamp("2000-01-01")   # start time
                        }
                        for i in range(seq_x.shape[0])
                    ]
                    # shape into GluonTS Dataset
                    gluonts_input = ListDataset(input_list, freq="1H")

                    # forecasting
                    predictor = self.model.create_predictor(batch_size=32)
                    outputs = list(predictor.predict(gluonts_input))

                    outputs = [f.median for f in outputs]                  # list of (prediction_length,)
                    outputs = torch.tensor(outputs).unsqueeze(-1)        # shape: (32, prediction_length, 1)

                elif self.args.model == 'Timer':
                    seq_x = seq_x.reshape(bs * n_vars, seq_len).cpu()
                    # seq_x = seq_x.cpu().squeeze(-1)
                    outputs = self.model.generate(seq_x, max_new_tokens=self.args.pred_len)
                    outputs = outputs.unsqueeze(-1)

                elif self.args.model == 'Chronos':
                    seq_x = seq_x.reshape(bs * n_vars, seq_len).cpu()
                    # seq_x = seq_x.cpu().squeeze(-1)
                    outputs = self.model.predict(seq_x, self.args.pred_len)
                    outputs = outputs.mean(dim=1).unsqueeze(-1)

                elif self.args.model == 'Chronos-bolt':
                    seq_x = seq_x.reshape(bs * n_vars, seq_len).cpu()
                    # seq_x = seq_x.cpu().squeeze(-1)
                    outputs = self.model.predict(seq_x, self.args.pred_len)
                    outputs = outputs.mean(dim=1).unsqueeze(-1)
                elif self.args.model == 'TimesFM':
                    # Cut into multiples of 32
                    seq_x = seq_x[:, :96, :]
                    seq_x = seq_x.reshape(bs * n_vars, -1).cpu()
                    # seq_x = seq_x.cpu().squeeze(-1)
                    lst = [seq_x[i] for i in range(seq_x.shape[0])]
                    freq = np.full(seq_x.shape[0], 0).tolist()
                    outputs, _ = self.model.forecast(lst, freq)
                    outputs = torch.tensor(outputs, dtype=torch.float32).unsqueeze(-1).to('cuda')
                elif self.args.model == 'Time-MoE':
                    seq_x = seq_x.reshape(bs * n_vars, seq_len)
                    outputs = self.model.generate(seq_x, max_new_tokens=self.args.pred_len)
                    outputs = outputs[:, -self.args.pred_len:].unsqueeze(-1)
                else:
                    print("error model!")

                outputs = outputs.reshape(bs, self.args.pred_len, n_vars)
                outputs = outputs.detach().cpu()
                # seq_y = seq_y.detach().cpu()
                score = criterion(outputs, seq_y)
                score = score.reshape(bs, n_vars, self.args.pred_len, 1)
                score = score.mean(dim=1)
                score = score.numpy()
                lab_x = lab_x.numpy()
                test_scores.append(score)
                test_labels.append(lab_x) 
        init_scores = np.concatenate(init_scores, axis=0).reshape(-1)
        init_scores = np.array(init_scores)
        test_scores = np.concatenate(test_scores, axis=0).reshape(-1)
        test_scores = np.array(test_scores)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("Inference time: {}".format(time.time() - time_now))

        # spot = SPOT()
        # spot.fit(init_scores, test_scores)
        # spot.initialize()
        # t_spot = float(spot.extreme_quantile)
        # print([t_spot])
        
        print("start evaluation!")
        evaluator = Evaluator(gt, test_scores, folder_path)
        evaluator.evaluate(metrics=self.args.metric, affiliation=self.args.t)  
        # evaluator.evaluate(metrics=self.args.metric, f1_raw=self.args.t)  




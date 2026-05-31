import os
import torch
import numpy as np
import time
import torch.nn as nn
import re
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, LargeScheduler
from transformers.trainer_pt_utils import get_parameter_names
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from ts_ad_evaluation import Evaluator

class Exp_Anomaly_Detection_SEMPO(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_SEMPO, self).__init__(args)

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args).to(self.device)
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
       
        optim_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        model_optim = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=1e-8,
        )
        
        # if self.args.use_weight_decay:
        #     model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, 
        #                             weight_decay=self.args.weight_decay)
        # else:
        #     model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        return model_optim
    
    def _select_criterion(self, train=0):
        if train:
            criterion = nn.MSELoss(reduction='mean')
        else:
            criterion = nn.MSELoss(reduction='none')
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion, flag='pretrain'):
        total_loss = []
        total_count = []

        self.model.eval()
        with torch.no_grad():
            for i, (seq_x, lab_x, seq_y, lab_y) in enumerate(vali_loader):
                B, T, dims = seq_x.size()
                seq_x = seq_x.float().to(self.device)
                seq_y = seq_y.float().to(self.device)
                lab_x = lab_x.float().expand(-1, -1, dims).to(self.device)

                if flag == 'pretrain':
                    recons, _ = self.model(seq_x)
                    recons = recons.detach().cpu()
                    seq_x = seq_x.detach().cpu()
                    loss = criterion(recons, seq_x)
                elif flag == 'train':
                    preds, recons = self.model(seq_x)
                    recons = recons.detach().cpu()
                    preds = preds.detach().cpu()
                    seq_x = seq_x.detach().cpu()
                    seq_y = seq_y.detach().cpu()
                    loss = criterion(recons, seq_x) + criterion(preds, seq_y)
            
                total_loss.append(loss)
                total_count.append(seq_x.shape[0])
                
                torch.cuda.empty_cache()
                
        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def pretrain(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')  

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(train=1)

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters() if param.requires_grad))

        scheduler = LargeScheduler(self.args, model_optim)
        global_step = 0

        # cal anomaly_socres
        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0

            loss_val = torch.tensor(0.).to(self.device)
            count = torch.tensor(0.).to(self.device)

            self.model.train()
            epoch_time = time.time()

            for i, (seq_x, lab_x, seq_y, lab_y) in enumerate(train_loader): 
                iter_count += 1
                model_optim.zero_grad()

                B, T, dims = seq_x.size()
                seq_x = seq_x.float().to(self.device)
                lab_x = lab_x.float().expand(-1, -1, dims).to(self.device)
                recons, _ = self.model(seq_x)

                loss = criterion(recons,seq_x)

                loss_val += loss.item()
                count += 1

                if i % 100 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                scheduler.schedule_step(global_step)
                global_step += 1
                torch.cuda.empty_cache()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()


            vali_loss = self.vali(vali_data, vali_loader, criterion, flag='pretrain')
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='pretrain')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
    

    def train(self, setting, train=0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')  
  
        # if train:
        #     print('loading model')
        #     original_setting = setting
        #     setting = re.sub(r'_' + self.args.data + '_', '_Monash_ADD_', setting)
        #     setting = re.sub(r'_st' + self.args.stride + '_', '_st1_', setting)
        #     checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), weights_only=False)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     setting = original_setting


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(train=1)

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters() if param.requires_grad))
        scheduler = LargeScheduler(self.args, model_optim)
        global_step = 0

        # cal anomaly_socres
        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0.).to(self.device)
            count = torch.tensor(0.).to(self.device)

            self.model.train()
            epoch_time = time.time()

            for i, (seq_x, lab_x, seq_y, lab_y) in enumerate(train_loader): 
                iter_count += 1
                model_optim.zero_grad()
                B, T, dims = seq_x.size()
                seq_x = seq_x.float().to(self.device)
                seq_y = seq_y.float().to(self.device)
                lab_x = lab_x.float().expand(-1, -1, dims).to(self.device)
                preds, recons = self.model(seq_x)
    
                loss = criterion(preds, seq_y) + criterion(recons, seq_x)

                loss_val += loss.item()
                count += 1

                if i % 100 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                scheduler.schedule_step(global_step)
                global_step += 1
                torch.cuda.empty_cache()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion, flag='train')
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='train')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
    
    
    def test(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))

        test_data, test_loader = self._get_data(flag='test')
        init_data, init_loader = self._get_data(flag='init')  # For SPOT algorithm    

        if test:
            print('loading model')
            original_setting = setting
            setting = re.sub(r'_' + str(self.args.data) + '_', '_Monash_ADD_', setting)
            setting = re.sub(r'_st' + str(self.args.stride) + '_', '_st1_', setting)
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            setting = original_setting

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)

        criterion = self._select_criterion(train=0)

        test_labels = []
        test_scores = []
        init_scores = []
        self.model.eval()
        # cal anomaly_socres
        # compute inference time
        time_now = time.time()
        with torch.no_grad():
            for i, (seq_x, lab_x, seq_y, lab_y) in enumerate(init_loader): 
                seq_x = seq_x.float().to(self.device)
                preds, recons = self.model(seq_x)
                score = torch.mean(criterion(recons, seq_x), dim=-1)
                score = score.detach().cpu().numpy()
                init_scores.append(score)
            for i, (seq_x, lab_x, seq_y, lab_y) in enumerate(test_loader):
                B, T, dims = seq_x.size()
                seq_x = seq_x.float().to(self.device)
                lab_x = lab_x.float().expand(-1, -1, dims).to(self.device)
                preds, recons = self.model(seq_x)
                score = torch.mean(criterion(recons, seq_x), dim=-1)
                score = score.detach().cpu().numpy()
                lab_x = lab_x.detach().cpu().numpy()
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
        
        evaluator = Evaluator(gt, test_scores, folder_path)
        evaluator.evaluate(metrics=self.args.metric, affiliation=self.args.t)  




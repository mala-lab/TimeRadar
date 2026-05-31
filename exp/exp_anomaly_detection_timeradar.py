import os
import torch
import numpy as np
import time
import torch.nn as nn
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, LargeScheduler
from transformers.trainer_pt_utils import get_parameter_names
import torch.distributed as dist
from ts_ad_evaluation import Evaluator
from transformers import AutoModel
import re
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F



class Exp_Anomaly_Detection_TimeRadar(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_TimeRadar, self).__init__(args)

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args).to(self.device)
            # margin loss
            if not hasattr(model, "tau_n"):
                model.register_buffer("tau_n", torch.tensor(0.0, device=self.device))
                model.register_buffer("tau_a", torch.tensor(0.0, device=self.device))
                model.register_buffer("tau_inited", torch.tensor(False, device=self.device))
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
            # margin loss
            if not hasattr(model, "tau_n"):
                model.register_buffer("tau_n", torch.tensor(0.0, device=self.device))
                model.register_buffer("tau_a", torch.tensor(0.0, device=self.device))
                model.register_buffer("tau_inited", torch.tensor(False, device=self.device))
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
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def _global_mean(self, x):
        # compute mean
        if x.numel() == 0:
            s = torch.zeros(1, device=x.device, dtype=x.dtype)
            c = torch.zeros(1, device=x.device, dtype=x.dtype)
        else:
            s = x.detach().float().sum().view(1)
            c = torch.tensor([x.numel()], device=x.device, dtype=x.dtype)

        if self.args.use_multi_gpu:
            dist.all_reduce(s, op=dist.ReduceOp.SUM)
            dist.all_reduce(c, op=dist.ReduceOp.SUM)

        if c.item() == 0:
            return None
        else:
            return (s / c).view(())


    def _ema_update(self, normal, abnormal):
        module = self.model.module if self.args.use_multi_gpu else self.model

        tau_n = self._global_mean(normal)
        tau_a = self._global_mean(abnormal)

        # EMA update
        m = self.args.ema_momentum
        with torch.no_grad():
            if tau_n is not None:
                if not module.tau_inited:
                    module.tau_n.copy_(tau_n)
                else:
                    module.tau_n.lerp_(tau_n, 1 - m)

            if tau_a is not None:
                if not module.tau_inited:
                    module.tau_a.copy_(tau_a)
                else:
                    module.tau_a.lerp_(tau_a, 1 - m)

            if (tau_n is not None) and (tau_a is not None):
                module.tau_inited.fill_(True)
        
        margin_loss = torch.tensor(0., device=self.device)
        if bool(module.tau_inited):
            margin_loss = (self.args.gamma - (module.tau_n - module.tau_a)).clamp_min(0.)
        return margin_loss
    

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_count = []

        module = self.model.module if self.args.use_multi_gpu else self.model

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                repr, outputs = self.model(batch_x, norm=self.args.norm)
                recon_loss = criterion(outputs, batch_x)
              
                B, T, dims = batch_x.size()
                BCD, patch_num, d_model = repr.size()
                # add margin loss in hidden space
                margin_loss = torch.tensor(0., device=self.device)
                if bool(module.tau_inited):
                    margin_loss = (self.args.gamma - (module.tau_n - module.tau_a)).clamp_min(0.)
                loss = recon_loss + self.args.margin_weight * margin_loss
                

                total_loss.append(loss.detach().item())
                total_count.append(batch_x.shape[0])
                
                # torch.cuda.empty_cache()
                
        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def finetuning(self, setting, train=0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')  

        if train:
            print('loading model')
            original_setting = setting
            setting = re.sub(r'_' + self.args.data + '_', '_Monash_ADD_', setting)
            setting = re.sub(r'_st' + str(self.args.stride) + '_', '_st1_', setting)
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), weights_only=False)
            self.model.load_state_dict(checkpoint)
            setting = original_setting

            # only finetuning reconstruction head and frozen others
            for param in self.model.module.parameters():
                param.requires_grad = False  
            for param in self.model.module.decoder.parameters():
                param.requires_grad = True

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters() if param.requires_grad))

        scheduler = LargeScheduler(self.args, model_optim)
        global_step = 0

        # cal anomaly_socres
        for epoch in range(self.args.finetune_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader): 
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                repr, outputs = self.model(batch_x, norm=self.args.norm)
                recon_loss = criterion(outputs, batch_x)
            
                B, T, dims = batch_x.size()
                BCD, patch_num, d_model = repr.size()

                # add margin loss in hidden space
                repr = torch.abs(repr).view(B, -1, dims, patch_num, d_model).mean(dim=1).squeeze()      # [bs * nvars x patch_num x d_model]
                repr = repr.view(-1, patch_num, d_model)
                repr = F.normalize(repr, p=2, dim=-1, eps=1e-12)      # [bs * nvars x patch_num x d_model]
                batch_y = batch_y.squeeze().view(B, patch_num, self.args.patch_len).amax(dim=2).bool()   # [bs x patch_num]   
                mask = batch_y.repeat_interleave(dims, dim=0)      # [bs * nvars x patch_num] 
                affinity = torch.matmul(repr, repr.transpose(1, 2))       # [bs * vars x patch_num x patch_num]
                affinity = (affinity.sum(dim=-1) - affinity.diagonal(dim1=1, dim2=2)) / (patch_num - 1) # [bs * nvars x patch_num]
                abnormal_affinity = affinity[mask] # [an_num]
                normal_affinity = affinity[~mask]  # [n_num]
                margin_loss = self._ema_update(normal_affinity, abnormal_affinity)
                loss = recon_loss + self.args.margin_weight * margin_loss
                
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
                # torch.cuda.empty_cache()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion)
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
    

    def train(self, setting):
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
        criterion = self._select_criterion()

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

            for i, (batch_x, batch_y) in enumerate(train_loader): 
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                repr, outputs = self.model(batch_x, norm=self.args.norm)
                recon_loss = criterion(outputs, batch_x)

                B, T, dims = batch_x.size()
                BCD, patch_num, d_model = repr.size()

                # add margin loss in hidden space
                repr = torch.abs(repr).view(B, -1, dims, patch_num, d_model).mean(dim=1).squeeze()      # [bs * nvars x patch_num x d_model]
                # normalize on embedding dim -> cosine similarity
                repr = F.normalize(repr, p=2, dim=-1, eps=1e-12)      # [bs * nvars x patch_num x d_model]
                batch_y = batch_y.squeeze().view(B, patch_num, self.args.patch_len).amax(dim=2).bool()   # [bs x patch_num]   
                mask = batch_y.repeat_interleave(dims, dim=0)      # [bs * nvars x patch_num] 
                affinity = torch.matmul(repr, repr.transpose(1, 2))       # [bs * nvars x patch_num x patch_num]
                affinity = (affinity.sum(dim=-1) - affinity.diagonal(dim1=1, dim2=2)) / (patch_num - 1) # [bs * nvars x patch_num]
                abnormal_affinity = affinity[mask] # [an_num]
                normal_affinity = affinity[~mask]  # [n_num]
                   

                margin_loss = self._ema_update(normal_affinity, abnormal_affinity)
                loss = recon_loss + self.args.margin_weight * margin_loss
                
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
                # torch.cuda.empty_cache()
        
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion)
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
            # self.model = self.model.module if hasattr(self.model, "module") else self.model
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
            setting = original_setting
            

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)

        criterion = self._select_criterion()

        test_labels = []
        test_scores = []
        init_scores = []
        self.model.eval()

        # cal anomaly_socres
        # compute inference time
        time_now = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(init_loader): 
                batch_x = batch_x.float().to(self.device)
                # score = self.model.module.infer(batch_x, norm=self.args.norm)
                repr, out_copies = self.model(batch_x, norm=self.args.norm)
                score = self.model.module.cal_anomaly_score(batch_x=batch_x, batch_out_copies=out_copies, anomaly_criterion=criterion)
                score = score.detach().cpu().numpy()
                init_scores.append(score)
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # score, repr = self.model.module.infer(batch_x, norm=self.args.norm, return_repr=True)
                repr, out_copies = self.model(batch_x, norm=self.args.norm)
                score = self.model.module.cal_anomaly_score(batch_x=batch_x, batch_out_copies=out_copies, anomaly_criterion=criterion)
                
                score = score.detach().cpu().numpy()
                test_scores.append(score)
                test_labels.append(batch_y) 

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
        # evaluator.evaluate(metrics=self.args.metric, f1_raw=self.args.t) 

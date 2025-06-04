from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from utils.dtw_metric import dtw,accelerated_dtw
from robust_loss_pytorch import AdaptiveLossFunction
import csv
import random
import numpy as np
import subprocess

import pandas
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
global_x = None
warnings.filterwarnings('ignore')
<<<<<<< HEAD

def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
    lines = result.stdout.split('\n')
    for line in lines:
        if 'python' in line:
            parts = line.split()
            if len(parts) >= 5:
                usage_str = parts[-2]
                try:
                    usage_mib = float(usage_str.replace('MiB', ''))
                    return usage_mib
                except ValueError:
                    pass
    return None

=======
>>>>>>> update
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.training_time = 0
<<<<<<< HEAD
        self.use_adaptiveloss = args.use_adaptiveloss
=======
>>>>>>> update

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

<<<<<<< HEAD
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 ** 2)
        print(f'Total number of parameters: {total_params}')
        print(f'Total size of parameters: {total_size_mb:.2f} MB')
        return total_size_mb

=======
>>>>>>> update
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

<<<<<<< HEAD
        self.print_model_size()
=======
>>>>>>> update
        start_time = time.time()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

<<<<<<< HEAD
        time_now = time.time()

        memory_usage_list = []
        time_per_iter_list = []
        max_memory_allocated = 0
=======
        time_per_iter_list = []
>>>>>>> update

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

<<<<<<< HEAD
        if self.use_adaptiveloss:
            adaptive = AdaptiveLossFunction(1, torch.float32, self.device, alpha_hi=3.0)
            criterion_tmp = adaptive.lossfun
            adaptive_optim = optim.AdamW(list(adaptive.parameters()), lr=0.001)
=======

        adaptive = AdaptiveLossFunction(1, torch.float32, self.device, alpha_hi=3.0)
        criterion_tmp = adaptive.lossfun
        adaptive_optim = optim.AdamW(list(adaptive.parameters()), lr=0.001)
>>>>>>> update

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

<<<<<<< HEAD
        # 打印模型结构
        print("Model Structure:")
        print(self.model)

        for epoch in range(self.args.train_epochs):

            if self.use_adaptiveloss:
                adaptive.print()
=======
        for epoch in range(self.args.train_epochs):

            adaptive.print()
>>>>>>> update
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_start_time = time.time()

                iter_count += 1
                model_optim.zero_grad()

<<<<<<< HEAD
                if self.use_adaptiveloss:
                    adaptive_optim.zero_grad()
=======
                adaptive_optim.zero_grad()
>>>>>>> update

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

<<<<<<< HEAD
                    if self.use_adaptiveloss:
                        loss = criterion_tmp((outputs - batch_y).flatten().unsqueeze(-1))
                    else:
                        loss = criterion(outputs, batch_y)
=======

                    loss = criterion_tmp((outputs - batch_y).flatten().unsqueeze(-1))

>>>>>>> update

                    loss = loss.mean()
                    train_loss.append(loss.item())

                iter_end_time = time.time()
                iter_duration = iter_end_time - iter_start_time
                time_per_iter_list.append(iter_duration)


<<<<<<< HEAD
                current_memory_allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)  # MiB
                memory_usage_list.append(current_memory_allocated)
                max_memory_allocated = max(max_memory_allocated, current_memory_allocated)

=======
>>>>>>> update
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = np.mean(time_per_iter_list[-100:])
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
<<<<<<< HEAD
                    if self.use_adaptiveloss:
                        adaptive_optim.step()
=======
                    adaptive_optim.step()
>>>>>>> update

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.training_time = time.time() - start_time

        return self.model

    def test(self, setting, test=0):
        global global_x
        torch.cuda.reset_max_memory_allocated(self.device)
<<<<<<< HEAD

        print("测试开始前的 GPU memory 使用情况：")
        initial_gpu_memory = get_gpu_memory_usage()
        print(f"初始 GPU memory 使用情况: {initial_gpu_memory} MiB")

        initial_memory = torch.cuda.memory_allocated(self.device) / (1024 * 1024)  # MiB
        print("初始内存: ", initial_memory)

=======
>>>>>>> update
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
<<<<<<< HEAD
        model_total_params = self.print_model_size()
=======
>>>>>>> update
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        batch_times = []
        with torch.no_grad():
            test_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_start_time = time.time()
                global_x=i
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                if self.args.use_amp:

                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
<<<<<<< HEAD
                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                    # 将 gt 和 pd 保存为 CSV 文件
                    df = pandas.DataFrame({'Ground Truth': gt, 'Prediction': pd})
                    csv_file_path = os.path.join(folder_path, str(i) + '.csv')
                    df.to_csv(csv_file_path, index=False)

            test_end_time = time.time()
            test_time_cost = test_end_time - test_time
=======
                # if i % 1 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                #
                #     # 将 gt 和 pd 保存为 CSV 文件
                #     df = pandas.DataFrame({'Ground Truth': gt, 'Prediction': pd})
                #     csv_file_path = os.path.join(folder_path, str(i) + '.csv')
                #     df.to_csv(csv_file_path, index=False)
>>>>>>> update
            avg_batch_time = 1000 * sum(batch_times) / len(batch_times)
            print(f"Average batch time: {avg_batch_time:.4f} ms")


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999

<<<<<<< HEAD

        mae, mse, rmse, mape,R2, mspe,nrmse,new_R2 = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{},R2:{}, mspe:{}, nrmse:{},new_R2:{},model_total_params:{},test_time_cost:{}'.format(mse, mae, rmse, mape,R2, mspe,nrmse,new_R2,model_total_params,test_time_cost))
=======
        mae, mse,nrmse,R2 = metric(preds, trues)
        print('mse:{}, mae:{}, nrmse:{},R2:{}'.format(mse, mae,nrmse,R2))
>>>>>>> update
        with open("result_long_term_forecast.csv", 'a', newline='') as f:

            writer = csv.writer(f)

            if f.tell() == 0:
<<<<<<< HEAD
                writer.writerow(["setting", "mse", "mae", "rmse", "mape","R2" ,"mspe","model_total_params","training_time","nrmse","new_R2","test_time_cost"])
            writer.writerow([self.args.model_id, mse, mae, rmse, mape, R2,mspe,model_total_params,self.training_time,nrmse,new_R2,test_time_cost])

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, R2,mspe,model_total_params,self.training_time,nrmse,new_R2,test_time_cost]))
=======
                writer.writerow(["setting", "mse", "mae", "nrmse","R2"])
            writer.writerow([self.args.model_id, mse, mae,nrmse,R2])

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, nrmse,R2]))
>>>>>>> update
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import math

from tqdm import tqdm

RECORDS_PER_EPOCH = 10

class Learner():
    def __init__(self, model, optimizer, metrics, train_loader, val_loader=None, save_path=''):
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.epoch_idx = 0
        self.metrics_hist = dict()
        for k in metrics.keys():
            self.metrics_hist[k] = []
            self.metrics_hist['val_'+k] = []
        self.m_loss = float('inf')
        self.act_mean_hist = defaultdict(list)
        for name, layer in model.named_children():
            print(name)
            layer.register_forward_hook(
                lambda layer, _, output: self.act_mean_hist[name].append(output.mean().item()) 
            )
            
    
    def run(self, epochs: int):
        for epoch in tqdm(range(epochs)):
            self.one_epoch()
        torch.save(self.model.state_dict(), Path(self.save_path,'last-parameters.pt'))

    def one_epoch(self):
        self.model.train()
        avg_metric = self._epoch_main_loop()
        self.model.eval()
        val_avg_metric = self._epoch_main_loop(eval=True)
        self._logger(avg_metric, val_avg_metric)
        self.epoch_idx += 1
        loss = val_avg_metric['val_loss'] if val_avg_metric else avg_metric['loss']
        if loss < self.m_loss:
            torch.save(self.model.state_dict(), Path(self.save_path,'best-parameters.pt'))

    def plot_act_mean(self):
        print(self.act_mean_hist)
        for key, hist in self.act_mean_hist.items():
            x = range(len(self.act_mean_hist[key]))
            print(hist)
            plt.plot(x, hist, label=key)
        plt.legend()
        plt.show()

    def plot(self, metric=None, n_row=None, n_col=None, **kwargs):
        print(self.metrics_hist)
        plt.figure(**kwargs)
        if metric == None:
            keys = self.metrics_hist.keys()
            if n_row:
                n_col = len(keys) // n_row
            if n_col:
                n_row = len(keys) // n_col
            else:
                n_row = math.floor(math.sqrt(len(keys)))
                n_col = math.ceil(len(keys)/n_row)

            plt.subplots_adjust(hspace=0.5)
            for i, key in enumerate(keys):
                plt.subplot(n_row, n_col, i+1)
                hist = self.metrics_hist[key]
                plt.title(key)
                plt.plot(range(len(hist)), hist)
        else:
            hist = self.metrics_hist[metric]
            plt.title(metric)
            plt.plot(range(len(hist)), hist)

    def _epoch_main_loop(self, eval=False):
        # set correct loader and return empty if the there isn't validation data
        loader = self.train_loader if not eval else self.val_loader
        metric_prefix = '' if not eval else 'val_'
        if not loader:
            return {}
        

        # create the data structures that will be used for recording batch run values
        metric_log = {metric_prefix+k:[] for k in self.metrics.keys()}
        size_log = []
        epoch_size = len(iter(loader))

        # record the metric value and batch size of every batch in order to calculate the epoch average
        for batch_idx, (X, y) in enumerate(loader):
            m = self._one_batch(X, y, eval=eval)
            for k, v in m.items():
                metric_log[metric_prefix+k].append(v.item())
                if batch_idx % (epoch_size // RECORDS_PER_EPOCH + 1) == 0:
                    self.metrics_hist[metric_prefix+k].append(v.item())
            size_log.append(len(X))

        # average the batch run values to get the epoch metric values
        avg_metric = {metric_prefix+k:0 for k in self.metrics.keys()}
        for k, values in metric_log.items():
            avg_metric[k] = sum([value*size for value, size in zip(values, size_log)]) / sum(size_log)
        
        return avg_metric

    def _one_batch(self, X, y, eval=False):
        output = self.model(X)
        loss = self.metrics['loss'](output, y)
        if not eval:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {name: func(output, y) for name, func in self.metrics.items()}
    

    def _logger(self, avg_metric, val_avg_metric):
        log = [f"{k}: {v:.4f}" for k, v in avg_metric.items()]
        val_log = [f"{k}: {v:.4f}" for k, v in val_avg_metric.items()]
        print(f"epoch: {self.epoch_idx}", *log, *val_log, sep=' | ')
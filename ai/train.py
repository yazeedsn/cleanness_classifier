from pathlib import Path
import matplotlib.pyplot as plt
import torch

class Learner():
    def __init__(self, model, optimizer, metrics, train_loader, val_loader=None, save_path=''):
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.epoch_idx = 0
        self.m_loss = float('inf')
    
    def run(self, epochs: int):
        for epoch in range(epochs):
            self.one_epoch()
        torch.save(self.model.state_dict(), Path(self.save_path,'last-parameters.pt'))

    def one_epoch(self):
        self.model.train()
        avg_metric = self._epoch_main_loop()
        self.model.eval()
        val_avg_metric = self._epoch_main_loop(eval=True)
        self._logger(avg_metric, val_avg_metric)
        self.epoch_idx += 1
        loss = val_avg_metric['loss'] if val_avg_metric else avg_metric['loss']
        if loss < self.m_loss:
            torch.save(self.model.state_dict(), Path(self.save_path,'best-parameters.pt'))
        plt.plot(self.epoch_idx, avg_metric['loss'].detach().numpy())
        plt.show()

    def _epoch_main_loop(self, eval=False):
        # set correct loader and return empty if the there isn't validation data
        loader = self.train_loader if not eval else self.val_loader
        if not loader:
            return {}
        
        # create the data structures that will be used for recording batch run values
        metric_log = {k:[] for k in self.metrics.keys()}
        size_log = []

        # record the metric value and batch size of every batch in order to calculate the epoch average
        for X, y in loader:
            m = self._one_batch(X, y, eval=eval)
            for k, v in m.items():
                metric_log[k].append(v)
            size_log.append(len(X))

        # average the batch run values to get the epoch metric values
        avg_metric = {k:0 for k in self.metrics.keys()}
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
        val_log = [f"val_{k}: {v:.4f}" for k, v in val_avg_metric.items()]
        print(f"epoch: {self.epoch_idx}", *log, *val_log, sep=' | ')
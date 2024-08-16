import torch
import numpy as np
import matplotlib.pyplot as plt

import os
from shutil import rmtree
from math import ceil


class Callback():
    def __init__(self):
        pass
    
class RecordMetricsCB(Callback):
    def __init__(self, metrics=None):
        super().__init__()
        self.metrics = metrics
        self.hist = {}

    def after_batch(self, _, phase, metric_value):
        for key, value in metric_value.items():
            if self.metrics and key not in self.metrics: continue
            if key not in self.hist: self.hist[key] = ([], [])
            if phase == 'train':
                self.hist[key][0].append(value)
            else:
                self.hist[key][1].append(value) 

    def clear(self):
        self.hist = {}


class LoggerCB(RecordMetricsCB):
    def __init__(self):
        super().__init__()

    def after_epoch(self, learner, phase):
        if phase == 'train': return
        avg = {}
        for key, value in self.hist.items():
            train_hist, valid_hist = value
            avg[f'{key}_train'] = sum(train_hist)/len(train_hist) 
            avg[f'{key}_valid'] = sum(valid_hist)/len(valid_hist)
        s = [f'Epoch: {learner.epoch}'] + [f'{k}: {v:.4}' for k,v in avg.items()]
        print(' | '.join(s))
        self.clear()


class ForwardHookCB(Callback):
    def __init__(self, hook):
        super().__init__()
        self.hook = hook
    
    def before_fit(self, learner):
        self.handlers = []
        for module in learner.model.modules():
            handler = module.register_forward_hook(self.hook)
            self.handlers.append(handler)

    def after_fit(self, _):
        for handler in self.handlers:
            handler.remove()


class MetricPlotterCB(RecordMetricsCB):
    def __init__(self, clear_hist=False, **plt_kw):
        super().__init__()
        self.clear_hist = clear_hist
        self.plt_kw = plt_kw

    def after_fit(self, learner):
        hist = self.hist
        ncols = min(2, len(hist))
        nrows = ceil(len(hist)/2)
        fig, axs = plt.subplots(nrows, ncols, **self.plt_kw)
        if len(axs.shape) == 1: 
            axs = np.expand_dims(axs, axis=0)
        r, c = 0, 0
        for k, (v_train, v_valid) in hist.items():
            x = list(range(len(v_train)))
            axs[r, c].plot(x, v_train)
            axs[r, c].plot(np.linspace(0, x[-1], len(v_valid)), v_valid)
            axs[r, c].set_title(k)
            c += 1 
            if c == ncols: 
                c = 0
                r += 1
        fig.legend(['Train','Valid'])
        if self.clear_hist: self.clear()

class TransformCB(Callback):
    def __init__(self, transform=None, target_transform=None, phase=None):
        super(TransformCB, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.phase = phase

    def before_batch(self, _, phase, batch):
        if (not self.phase) or (phase == self.phase):
            batch[0] = self.transform(batch[0]) if self.transform else batch[0]
            batch[1] = self.target_transform(batch[1]) if self.target_transform else batch[1]

class LRSchedulerCB(Callback):
    def __init__(self, scheduler):
        super(LRSchedulerCB, self).__init__()
        self.scheduler = scheduler

    def after_batch(self, _, phase, __):
        if phase == 'train':
            self.scheduler.step()


class LRFinderCB(Callback):
    def __init__(self, lr_scheduler, start_lr=1e-5, max_lr=10, smooth_f=0.05, break_f=3, xscale='log'):
        super(LRFinderCB, self).__init__()
        self.lr_scheduler = lr_scheduler
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.smooth_f = smooth_f
        self.break_f = break_f
        self.xscale = xscale
        self.best_loss = float('inf')
        self.hist = {'lr': [], 'loss': []}

    def before_fit(self, learner):
        self.temp_path = os.path.join(os.getcwd(),'.temp_lr_finder_cb')
        self.state_file_name = 'model_initial_state.pt'
        if not os.path.exists(self.temp_path): os.mkdir(self.temp_path)
        torch.save(learner.model.state_dict(), os.path.join(self.temp_path, self.state_file_name))
        for g in learner.optimizer.param_groups:
            g['lr'] = self.start_lr

    def after_batch(self, learner, phase, value):
        if phase != 'train': return
        loss = value['loss']
        self.best_loss = min(loss, self.best_loss)
        if self.hist['loss']:
            prev_loss = self.hist['loss'][-1]
            smooth_loss = (loss * (1-self.smooth_f) + prev_loss * self.smooth_f) / 2
        else:
            smooth_loss = loss
        self.hist['lr'].append(self.lr_scheduler.get_last_lr())
        self.hist['loss'].append(smooth_loss)
        if loss > self.break_f*self.best_loss or self.lr_scheduler.get_last_lr()[0] >= self.max_lr:
            learner.stop_fit_request()
        self.lr_scheduler.step()

    def after_fit(self, learner):
        model_state_dict = torch.load(os.path.join(self.temp_path, self.state_file_name))
        learner.model.load_state_dict(model_state_dict)
        rmtree(self.temp_path)
        self.plot()

    def plot(self):
        plt.xscale(self.xscale)
        plt.xlabel('lr')
        #plt.ylim([int(self.best_loss-0.5), int(self.best_loss+3)])
        plt.ylabel('loss')
        plt.plot(self.hist['lr'][1:], self.hist['loss'][1:])



class LSUVCB(Callback):
    def __init__(self, max_iter= 100, verbose=False):
        super(LSUVCB, self).__init__()
        self.filter = ['Conv2d', 'Linear']
        self.max_iter = max_iter
        self.verbose = verbose

    def on_init(self, learner):
        xb, _ = next(iter(learner.dataloaders['train']))
        for m in learner.model.modules():
            if type(m).__name__ not in self.filter: continue
            if not next(m.parameters()).requires_grad: continue
            if self.verbose: print(m)
            h = m.register_forward_hook(self._stats_hook)
            with torch.no_grad():
                i = 0
                while learner.model(xb) is not None and (abs(self.mean) > 1e-3 or abs(self.std-1) > 1e-3):
                    m.weight.data /= self.std
                    if m.bias is not None: m.bias -= self.mean
                    elif abs(self.mean) > 1e-3: break
                    i += 1
                    if i == self.max_iter: break
            h.remove()

    def _stats_hook(self, module, input, output):
        act = output.cpu().detach()
        self.mean = act.mean()
        self.std = act.std()
    
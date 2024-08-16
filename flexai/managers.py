import torch
import matplotlib.pyplot as plt

from math import ceil

class ActivationStatsManger():
    def __init__(self, model, type_filter=[]):
        self.model = model
        self.type_filter = type_filter
        self.attr_name = 'stat'

    def register_stats(self, module, input, output):
        if type(module).__name__ in self.type_filter:
            if not hasattr(module, self.attr_name): 
                setattr(module, self.attr_name, {'mean': [], 'std':[], 'histc':[]}) 
            if module.training:
                getattr(module, self.attr_name)['mean'].append(output.mean().item())
                getattr(module, self.attr_name)['std'].append(output.std().item())
                getattr(module, self.attr_name)['histc'].append(output.abs().histc(40, 0, 10))

    def get_stat(self, stat_name):
        stat = {}
        idx = 1
        for m in self.model.modules():
            if not hasattr(m, self.attr_name): continue
            stat[f'{type(m).__name__} {idx}'] = getattr(m, self.attr_name)[stat_name]
            idx += 1
        return stat
    
    def mean_std(self, **fig_kw):
        means, stds = self.get_stat('mean'), self.get_stat('std')
        fig, axs = plt.subplots(1, 2, **fig_kw)
        
        layers = means.keys()
        for layer in layers:
            axs[0].plot(means[layer])
            axs[1].plot(stds[layer])
        fig.legend(layers)
    
    def color_dim(self):
        histcs = self.get_stat('histc')
        hists = []
        names = []
        for name, histc in histcs.items():
            hist = torch.stack(histc).t().float().log1p()
            hists.append(hist)
            names.append(name)
        ncols = 2
        nrows = ceil(len(hists)/ncols)
        fig, axs = plt.subplots(nrows, ncols)
        fig.subplots_adjust(hspace=1)
        fig.set_figwidth(10)
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            ax.set_axis_off()
            if i >= len(hists): break
            ax.imshow(hists[i], origin='lower', aspect='auto')
            ax.set_title(names[i])
            

    def dead_chart(self):
        histcs = self.get_stat('histc')
        hists = []
        names = []
        for name, histc in histcs.items():
            hist = torch.stack(histc).t().float().log1p()
            hists.append(hist[0] / hist.sum(0))
            names.append(name)
        
        ncols = 2
        nrows = ceil(len(hists)/ncols)
        fig, axs = plt.subplots(nrows, ncols)
        fig.subplots_adjust(hspace=1)
        fig.set_figwidth(12)

        axs = axs.flatten()
        for i, ax in enumerate(axs):
            if i >= len(hists): 
                ax.set_axis_off()
                break
            ax.plot(hists[i])
            ax.set_title(names[i])
            ax.set_ylim(0, 1)
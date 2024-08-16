import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def show_batch(images, labels=None, idx_to_class=None, **fig_kw):
        ncols = min(8, len(images))
        nrows = ceil(len(images)/ncols)
        fig, axs = plt.subplots(nrows, ncols, **fig_kw)
        if len(images) == 1: axs = np.array([axs])
        axs = axs.flatten()
        for ax in axs:  ax.set_axis_off()        
        for i, image in enumerate(images):
                axs[i].imshow(image.mean(0), cmap='gray')
                if labels is not None:
                        label = idx_to_class[labels[i].item()] if idx_to_class else labels[i].item()
                        axs[i].set_title(label)
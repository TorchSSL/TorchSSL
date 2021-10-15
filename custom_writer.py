import datetime
from pathlib import Path
from typing import Sequence, Union, Tuple
import numpy as np
import json
import torch
import os

class CustomWriter(object):
    '''
    Custom Writer for training record.
    Parameters:
    -----------
    log_dir : pathlib.Path or str, path to save logs.
    enabled : bool, whether to enable tensorboard writer.
    '''
    def __init__(self, log_dir, enabled=True):
        self.writer = None
        self.selected_module = ''

        if enabled:
            self.log_dir = str(log_dir)
            self.stats = {}
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
        
        # Attributes to record
        self.epoch = 0
        self.mode = None
        self.timer = datetime.datetime.now()
        self.tb_writer_funcs = {
            'add_scalar', 'add_scalars',
            'add_image', 'add_images',
            'add_figure',
            'add_audio',
            'add_text',
            'add_histogram',
            'add_pr_curve',
            #'add_embedding', # TODO: problem with add_embedding
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'} # TODO : Test these two funcs.
    
    def dump_stats(self):
        with open(f"{self.log_dir}/log", "w") as f:
            json.dump(self.stats, f,
                indent=4,
                ensure_ascii=False,
                separators=(",", ": "),
        )
    
    def set_epoch(self, epoch, mode):
        '''
        Execute this function to update the step attribute and compute the cost time of one epoch in seconds.
        Recommend to run this function every step.
        This function MUST be executed before other custom writer functions.
        Parameters:
        ------------
        step : int, step number.
        mode : str, 'train' or 'valid'
        '''
        if epoch == 0:
            self.timer = datetime.datetime.now()
        elif epoch != self.epoch:
            duration = datetime.datetime.now() - self.timer
            second_per_epoch = duration.total_seconds() / (epoch - self.epoch)
            self.add_scalar(tag='second_per_epoch', data=second_per_epoch)
        self.epoch = epoch
        self.mode = mode

    def get_epoch(self) -> int:
        return self.epoch
    
    def get_keys(self, epoch: int = None) -> Tuple[str, ...]:
        """Returns keys1 e.g. train,eval."""
        if epoch is None:
            epoch = self.get_epoch()
        return tuple(self.stats[epoch])

    def get_keys2(self, key: str, epoch: int = None) -> Tuple[str, ...]:
        """Returns keys2 e.g. loss,acc."""
        if epoch is None:
            epoch = self.get_epoch()
        d = self.stats[epoch][key]
        keys2 = tuple(k for k in d if k not in ("time", "total_count"))
        return keys2
    
    def plot_stats(self):
        self.matplotlib_plot(self.log_dir)
    
    def matplotlib_plot(self, output_dir: Union[str, Path]):
        """Plot stats using Matplotlib and save images."""
        keys2 = set.union(*[set(self.get_keys2(k)) for k in self.get_keys()])
        for key2 in keys2:
            keys = [k for k in self.get_keys() if key2 in self.get_keys2(k)]
            plt = self._plot_stats(keys, key2)
            p = Path(output_dir) / f"{key2}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p)

    def _plot_stats(self, keys: Sequence[str], key2: str):
        # str is also Sequence[str]
        if isinstance(keys, str):
            raise TypeError(f"Input as [{keys}]")

        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        plt.clf()

        epochs = sorted(list(self.stats.keys()))
        for key in keys:
            y = [
                self.stats[e][key][key2]
                if e in self.stats
                and key in self.stats[e]
                and key2 in self.stats[e][key]
                else np.nan
                for e in epochs
            ]
            assert len(epochs) == len(y), "Bug?"

            plt.plot(epochs, y, label=key2, marker="x")
        plt.legend()
        plt.title(f"iteration vs {key2}")
        # Force integer tick for x-axis
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel("iteration")
        plt.ylabel(key2)
        plt.grid()
        return plt

    def to_numpy(self, a):
        if isinstance(a, list):
            return np.array(a)
        for kind in [torch.Tensor, torch.nn.Parameter]:
            if isinstance(a, kind):
                if hasattr(a, 'detach'):
                    a = a.detach()
                return a.cpu().numpy()
        return a
    
    def add_scalar(self, tag, data):
        data = self.to_numpy(data)
        data = float(data)
        self.stats.setdefault(self.epoch, {}).setdefault(self.mode, {})[tag] = data
    

    def __getattr__(self, name):
        if name in self.tb_writer_funcs:
            func = getattr(self, name, None)
            # Return a wrapper for all functions.
            def wrapper(tag, data, *args, **kwargs):
                if func is not None:
                    if name not in self.tag_mode_exceptions:
                        tag = f"{tag}/{self.mode}"
                    func(tag, data, *args, global_step=self.step, **kwargs)
    
            return wrapper
        else:
            # default __getattr__ function to get other attributes.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


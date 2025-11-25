
import pandas as pd
import logging.config
import logging
import torch.nn.functional as F
import torch
import shutil
from enum import Enum
import torch.distributed as dist
import os
import sys

class Results():

    def __init__(self):

        self.results = None

    def add(self, **kwargs):

        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, save_path):

        self.results.to_csv(save_path, index=False, index_label=False)


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass

        return no_op


def get_logger(log_dir, log_name='log.txt', resume=False, is_rank0=True):
    """Get the program logger.
    Args:
        log_dir (str): The directory to save the log file.
        log_name (str, optional): The log filename. If None, it will use the main
            filename with ``.log`` extension. Default is None.
        resume (str): If False, open the log file in writing and reading mode.
            Else, open the log file in appending and reading mode; Default is "".
        is_rank0 (boolean): If True, create the normal logger; If False, create the null
           logger, which is useful in DDP training. Default is True.
    """
    if is_rank0:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)

        # StreamHandler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)

        # FileHandler
        mode = "w+" if not resume else "a+"
        if log_name is None:
            log_name = os.path.basename(sys.argv[0]).split(".")[0] + (".log")
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode=mode)
        file_handler.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)
    else:
        logger = NoOp()

    return logger


def setup_logging(log_file='log.txt', log_flag=False):
    if log_flag:
        from loguru import logger
        logger.configure(handlers=[
                        dict(sink=sys.stdout, format='<level>{message}</level>', level='INFO', enqueue=True),
                        dict(sink=log_file, level='INFO', enqueue=True),
                        ])
        return logger
    return NoOp()


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filename = os.path.join(path, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_model = os.path.join(path, 'model_best.pth.tar')
        shutil.copyfile(filename, best_model)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class average_meter(object):
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.summary_type = summary_type
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class progress_meter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def display_summary(self, logger):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, lr_initial):
    lr = lr_initial * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weight_decay_sep(model, weight_decay, ignore_list=[]):
    pw = []
    pwo = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1 or name in ignore_list:
                pwo.append(param)
            else:
                pw.append(param)
    return [{'params': pwo, 'weight_decay': 0.},
            {'params': pw, 'weight_decay': weight_decay}]



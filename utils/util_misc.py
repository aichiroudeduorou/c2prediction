import numpy as np
import itertools
import random
import numpy as np
import torch
import omegaconf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def reproduc(seed, benchmark=False, deterministic=True):
    """Make experiments reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def omegaconf2list(opt, prefix='', sep='.'):
    notation_list = []
    for k, v in opt.items():
        k = str(k)
        if isinstance(v, omegaconf.listconfig.ListConfig):
            notation_list.append("{}{}={}".format(prefix, k, v))
        elif isinstance(v, (float, str, int,)):
            notation_list.append("{}{}={}".format(prefix, k, v))
        elif v is None:
            notation_list.append("{}{}=~".format(prefix, k,))
        elif isinstance(v, omegaconf.dictconfig.DictConfig):
            nested_flat_list = omegaconf2list(v, prefix + k + sep, sep=sep)
            if nested_flat_list:
                notation_list.extend(nested_flat_list)
        else:
            raise NotImplementedError
    return notation_list


def omegaconf2dotlist(opt, prefix='',):
    return omegaconf2list(opt, prefix, sep='.')


def omegaconf2dict(opt, sep):
    notation_list = omegaconf2list(opt, sep=sep)
    dict = {notation.split('=', maxsplit=1)[0]: notation.split(
        '=', maxsplit=1)[1] for notation in notation_list}
    return dict



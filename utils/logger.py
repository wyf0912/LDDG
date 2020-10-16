from collections import OrderedDict
import os
import re
import torch
import numpy as np
from tensorboardX import SummaryWriter

np.set_printoptions(precision=2)


class Logger:
    def __init__(self, file_path='', tensorboard_writer: SummaryWriter = None):
        self.writer = tensorboard_writer
        self.iter_cnt = 0
        self.iter_info = OrderedDict()
        self.file_path = file_path
        self.log_file = open(self.file_path, 'w')
        # self.print = print

    def collect_iter_info(self, info):
        self.iter_cnt += 1
        for key, val in info.items():
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            if key not in self.iter_info.keys():
                self.iter_info[key] = 0
            self.iter_info[key] += val

    def write_log(self, *log):  # , refresh=False
        print(*log)
        if self.log_file is not None:
            self.log_file.write(str(log) + '\n')
            # if refresh:
            self.log_file.close()
            self.log_file = open(self.file_path, 'a')

    def log_train_info(self, epoch):
        if self.log_file is not None:
            print = self.write_log
        print('\n\nEpoch:%d' % (epoch))
        for key, val in self.iter_info.items():
            self.iter_info[key] = val / self.iter_cnt
            if isinstance(self.iter_info[key], np.ndarray):
                print('\t%s:%s' % (key, str(self.iter_info[key])))
            else:
                print('\t%s:%.4f' % (key, self.iter_info[key]))
            if self.writer:
                if isinstance(self.iter_info[key], np.ndarray):
                    self.writer.add_histogram('train/' + key, val / self.iter_cnt, epoch)
                else:
                    self.writer.add_scalar('train/' + key, val / self.iter_cnt, epoch)
        print('\n\n')
        train_info_copy = self.iter_info.copy()
        self.iter_cnt = 0
        self.iter_info = OrderedDict()
        return train_info_copy

    def log_epoch_info(self, info_list, epoch):
        if self.log_file is not None:
            print = self.write_log
        print('\n\nEpoch:%d Valid' % (epoch))
        for info_dict in info_list:
            print('\t%s:%.6f' % (info_dict['name'], info_dict['val']))
            if self.writer:
                self.writer.add_scalar('eval/' + info_dict['name'], info_dict['val'], epoch)
        print('\n\n')
        return info_dict

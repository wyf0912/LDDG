import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import os
import re
import SimpleITK as stik
from collections import OrderedDict
from torchvision import transforms
import random
import cv2
import numpy as np
import torch
import SynchronousTransforms.transforms as T


def makeDataset(phase='train', path='/home/hlli/project/yufei/Medical_Segmentation_Dataset/GGM_spinal_cord_challenge',
                specific_domain=None, transform_train=None, transform_eval=None):
    """
    :param transform_train:
    :param phase: train or infer
        train: return slice, gt
        infer: return a
    :param path:
    :param specific_domain:
        None: return all domains
        list of str "site%d" : return specified one or several datasets
    :return:
    """
    assert phase in ['train', 'infer', 'train_nips']
    path1, path2 = os.path.join(path, 'train'), os.path.join(path, 'test')
    if phase == 'train' or 'train_nips':
        imageFileList = [os.path.join(path1, f) for f in os.listdir(path1) if 'site' in f and '.txt' not in f]
    elif phase == 'infer':
        imageFileList = [os.path.join(path2, f) for f in os.listdir(path2) if 'site' in f and '.txt' not in f]
    data_dict = {'site1': OrderedDict(), 'site2': OrderedDict(), 'site3': OrderedDict(), 'site4': OrderedDict()}
    for file in imageFileList:
        res = re.search('site(\d)-sc(\d*)-(image|mask)', file).groups()
        if res[1] not in data_dict['site' + res[0]].keys():
            data_dict['site' + res[0]][res[1]] = {'input': None, 'gt': []}
        if res[2] == 'image':
            data_dict['site' + res[0]][res[1]]['input'] = file
        if res[2] == 'mask':
            data_dict['site' + res[0]][res[1]]['gt'].append(file)
    datasets = {}
    print('Making dataset...')
    resolution = {
        'site1': [5, 0.5, 0.5],
        'site2': [5, 0.5, 0.5],
        'site3': [2.5, 0.5, 0.5],
        'site4': [5, 0.29, 0.29],
    }
    for domain, data_list in data_dict.items():
        if specific_domain is None or domain in specific_domain:
            datasets[domain] = SpinalCordDataset(data_list, phase=phase, transform_train=transform_train,
                                                 transform_eval=transform_eval, resolution=resolution[domain])
    print('Dataset finished')
    return datasets


class SpinalCordDataset(dataset.Dataset):
    def __init__(self, data_list, phase, transform_train=None, transform_eval=None, **kwargs):
        self.phase = phase
        self.reader = stik.ImageFileReader()
        self.reader.SetImageIO("NiftiImageIO")
        self.data_list = self.__read_dataset_into_memory(data_list)
        self.map_list = self.__get_index_map()
        self.info_dict = kwargs

        self.real_sample_num = len(self.data_list) if phase == 'infer' else len(self.map_list)

        self.input_transform = None  # transforms.Compose(transforms.ToPILImage,transforms.ToTensor)
        self.gt_transform = None
        # self.transform_train = T.ComposedTransform()
        if transform_train is None:
            transform_train = T.ComposedTransform([T.RandomCrop(160), T.CenterCrop(144)])  # 144 y  128 n # ,
        if transform_eval is None:
            transform_eval = T.ComposedTransform([T.CenterCrop(144)])  # 没有用目前
        self.transform_train = transform_train

    def __get_index_map(self):
        map_list = []
        total_slice_num = 0
        for data in self.data_list.values():
            slice_num = data['input'].shape[0]
            for i in range(slice_num):
                map_list.append([data['input'][i], np.stack([data['gt'][idx][i] for idx in range(4)], axis=0)])
            total_slice_num += slice_num
        return map_list

    def __read_dataset_into_memory(self, data_list):
        for val in data_list.values():
            val['input'] = self.read_numpy(val['input'])
            for idx, gt in enumerate(val['gt']):
                val['gt'][idx] = self.read_numpy(gt)
        return data_list

    def __getitem__(self, idx):
        if self.phase == 'train' or self.phase == 'train_nips':
            if self.phase == 'train_nips':
                idx = random.randint(0, self.real_sample_num - 1)
            x, gt_list = self.map_list[idx]
            x = x / (x.max() if x.max() > 0 else 1)
            gt_list = torch.tensor(gt_list, dtype=torch.uint8)
            spinal_cord_mask = (torch.mean((gt_list > 0).float(), dim=0) > 0.5).float()
            gm_mask = (torch.mean((gt_list == 1).float(), dim=0) > 0.5).float()
            # a1 = [torch.sum(spinal_cord_mask), torch.sum(gm_mask)]
            x, spinal_cord_mask, gm_mask = self.transform_train(x, spinal_cord_mask, gm_mask)
            # a2 = [torch.sum(spinal_cord_mask), torch.sum(gm_mask)]
            # if a1 != a2:
            #     print(a1, a2)
            return x, spinal_cord_mask, gm_mask
        elif self.phase == 'infer':
            list_temp = list(self.data_list.values())[idx]
            x, gt_list = list_temp['input'], list_temp['gt']
            return x, gt_list

    def __len__(self):
        if self.phase == 'train' or self.phase == 'train_nips':
            return len(self.map_list)
        elif self.phase == 'infer':
            return len(self.data_list)
        # elif self.phase == 'train_nips':
        #     return self.train_nips_sample_num

    def read_numpy(self, file_name):
        self.reader.SetFileName(file_name)
        data = self.reader.Execute()
        return stik.GetArrayFromImage(data)

    def set_phase(self, phase):
        assert phase in ['train', 'test', 'infer']


if __name__ == '__main__':
    datasets = makeDataset()
    for d in datasets.values():
        sample_num = len(d)
        for i in range(sample_num):
            d[i]
    # dataloader.DataLoader()

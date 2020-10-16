from torchvision import transforms
import cv2
from SynchronousTransforms import Rand
import numbers
import torch.nn.functional as F
import numpy as np
import random
import random
from functools import wraps
import torch
from torchvision import transforms


def pre_post_deal(func):
    """
    The wrap converts the tensor into numpy and reshape the input from 1*1*w*h to w*h
    after transform, convert the output to the tenosr with 1*1*w*h
    :param func:
    :return: tensor
    """

    @wraps(func)
    def checked(self, img, gt, gt2):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if isinstance(gt2, torch.Tensor):
            gt2 = gt2.detach().cpu().numpy()
        img = img.squeeze()
        gt = gt.squeeze()
        gt2 = gt2.squeeze()
        if random.random() > 0.5 or func.__qualname__ == 'RandomCrop.__call__' or func.__qualname__ == 'CenterCrop.__call__':
            img_, gt_, gt2_ = func(self, img, gt, gt2)
        else:
            img_, gt_, gt2_ = img, gt, gt2
        shape = gt_.shape
        return torch.tensor(img_).reshape([1, *shape]).float(), torch.tensor(gt_).reshape(
            [1, *shape]).float(), torch.tensor(gt2_).reshape(
            [1, *shape]).float()

    return checked


syn_rand = Rand()


def pad(img, val):
    """
    :param val: int or list with length of 4 (w1,w2,h1,h2)
    :return: padded img
    """
    if isinstance(val, numbers.Number):
        val = [val for _ in range(4)]
    assert len(img.shape) in [2]
    w, h = img.shape
    new_img = np.zeros([w + val[0] + val[1], h + val[2] + val[3]])
    new_img[val[0]:w + val[0], val[2]:h + val[2]] = img
    return new_img


class RandomCrop:
    def __init__(self, size=224):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def get_params(self, img):
        w, h = img.shape
        th, tw = self.size  # int(h * scale), int(w * scale)
        if w <= tw:  # and h <= th:
            tw = w
        if h <= th:
            th = h
        # return 0, 0, h, w

        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)
        # print(w, h, i, j, th, tw)
        return i, j, th, tw

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        h, w = self.size
        img_h, img_w = img.shape
        pad_val = [0, 0, 0, 0]
        if h > img_h:
            pad_val[0] = (h - img_h) // 2
            pad_val[1] = (h - img_h) - pad_val[0]
        if w > img_w:
            pad_val[2] = (w - img_w) // 2
            pad_val[3] = (w - img_w) - pad_val[2]
        if sum(pad_val) != 0:
            img = pad(img, pad_val)
            gt = pad(gt, pad_val)
            gt2 = pad(gt2, pad_val)
        i, j, h, w = self.get_params(img)
        return img[i:i + h, j:j + w], gt[i:i + h, j:j + w], gt2[i:i + h, j:j + w]


class CenterCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        h, w = self.size
        img_h, img_w = img.shape
        pad_val = [0, 0, 0, 0]
        if h > img_h:
            pad_val[0] = (h - img_h) // 2
            pad_val[1] = (h - img_h) - pad_val[0]
        if w > img_w:
            pad_val[2] = (w - img_w) // 2
            pad_val[3] = (w - img_w) - pad_val[2]
        if sum(pad_val) != 0:
            img = pad(img, pad_val)
            gt = pad(gt, pad_val)
            gt2 = pad(gt2, pad_val)
        img_h, img_w = img.shape
        i_0, i_1 = img_h // 2 - h // 2, img_h // 2 + h // 2
        j_0, j_1 = img_w // 2 - w // 2, img_w // 2 + w // 2
        return img[i_0:i_1, j_0:j_1], gt[i_0:i_1, j_0:j_1], gt2[i_0:i_1, j_0:j_1]


class Sharpness:
    def __init__(self, strength_range=None):
        if strength_range is None:
            strength_range = [10, 30]
        self.strength_range = strength_range

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        center_val = random.uniform(*self.strength_range)
        kernel = np.ones([3, 3], dtype=np.float32) * (-(center_val - 1) / 8)
        kernel[1, 1] = center_val
        out = cv2.filter2D(img, kernel=kernel, ddepth=-1)
        return out, gt, gt2


class Blurriness:
    def __init__(self, ksize=3, sigma_range=None):
        if sigma_range is None:
            sigma_range = [0.25, 1.5]
        self.ksize = ksize
        self.sigma_range = sigma_range

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        out = cv2.GaussianBlur(img, ksize=(self.ksize, self.ksize), sigmaX=random.uniform(*self.sigma_range))
        return out, gt, gt2


class Noise:
    def __init__(self, std_range=None):
        if std_range is None:
            std_range = [0.1, 1.0]
        self.std_range = std_range

    def add_gaussian_noise(self, img, std, mean=0):
        noise = np.random.normal(mean, std, img.shape)
        out = noise + img
        out = np.clip(out, a_min=0, a_max=1)
        return out

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        out = self.add_gaussian_noise(img, random.uniform(*self.std_range))
        return out, gt, gt2


class Brightness:
    def __init__(self, scale_shift_range=None, bias_range=None):
        if bias_range is None:
            bias_range = [-0.1, 0.1]
        self.bias_range = bias_range
        if scale_shift_range is None:
            scale_shift_range = [-0.1, 0.1]
        self.scale_shift_range = scale_shift_range

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        out = img * (1 + random.uniform(*self.scale_shift_range)) + random.uniform(*self.bias_range)
        out = np.clip(out, a_min=0, a_max=1)
        return out, gt, gt2


class Rotation:
    def __init__(self, angle_range=None):
        if angle_range is None:
            angle_range = [-20, 20]
        self.angle_range = angle_range

    @staticmethod
    def rotate(image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        random_angle = random.uniform(*self.angle_range)
        out = Rotation.rotate(img, angle=random_angle)
        gt_ = Rotation.rotate(gt, angle=random_angle)
        gt2_ = Rotation.rotate(gt2, angle=random_angle)
        return out, gt_, gt2_


class Scale:
    def __init__(self, magnitude_range=None):
        if magnitude_range is None:
            magnitude_range = [0.4, 1.6]
        self.magnitude_range = magnitude_range

    @pre_post_deal
    def __call__(self, img, gt, gt2):
        scale_x = random.uniform(*self.magnitude_range)
        scale_y = random.uniform(*self.magnitude_range)
        img_ = cv2.resize(img, dsize=(0, 0), fx=scale_x, fy=scale_y)
        gt_ = cv2.resize(gt, dsize=(0, 0), fx=scale_x, fy=scale_y)
        gt2_ = cv2.resize(gt2, dsize=(0, 0), fx=scale_x, fy=scale_y)
        return img_, gt_, gt2_


class ComposedTransform:
    def __init__(self, transform_list=None):
        if transform_list is None:
            transform_list = [CenterCrop(160), Sharpness([0, 30]), Blurriness(), Noise([0., 0.05]), Brightness(),
                              Rotation(), Scale([0.7, 1.3]), RandomCrop(144)]
        self.transform_list = transform_list

    def __call__(self, img, gt, gt2):
        for transform in self.transform_list:
            img, gt, gt2 = transform(img, gt, gt2)
        return img, gt, gt2

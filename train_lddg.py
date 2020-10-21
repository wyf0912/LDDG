import argparse
import logging
import os
from dataset import spinal_cord_challenge
import sys
import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
# from eval import eval_net
from unet import unet
from tensorboardX import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader, random_split
from utils import get_box, get_center
import utils.triplet
import utils.logger
import utils
from torch.autograd import Function
import eval
import cv2
from torchvision.utils import save_image
import torch.nn.functional as F
import SynchronousTransforms.transforms as T
from SynchronousTransforms import transforms

# import segmentation_models_pytorch as smp

gpu_id = utils.get_available_GPUs(1, 1., 0.4)[0]
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
print("GPU_ID:%d" % gpu_id)

eps = 1e-10


# Ours, need reparametric trick!
def kl_gaussian_loss(mu, logvar):
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss


# faster convolutions, but more memory
# cudnn.benchmark = True
def get_batch(dataset, batch_size):
    x_list = []
    spinal_mask_list = []
    gm_mask_list = []
    for _ in range(batch_size):
        x, spinal_cord_mask, gm_mask = dataset[0]
        x_list.append(x)
        spinal_mask_list.append(spinal_cord_mask)
        gm_mask_list.append(gm_mask)
    return x_list, spinal_mask_list, gm_mask_list
    # return torch.stack(x_list, dim=0).cuda(), torch.stack(spinal_mask_list, dim=0).cuda(), torch.stack(gm_mask_list,dim=0).cuda()


def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()


class LowRank(Function):
    @staticmethod
    def forward(ctx, x):
        U, S, V = torch.svd(x)
        ctx.save_for_backward(x, U, V)
        return torch.sum(S)

    @staticmethod
    def backward(ctx, grad_output):
        data = ctx.saved_tensors
        grad = torch.mm(data[1], data[2].t())
        return grad_output * grad


def get_multi_batch(dataset_list, batch_size):
    x_list = []
    spinal_mask_list = []
    gm_mask_list = []
    for dataset in dataset_list:
        x, spinal_cord_mask, gm_mask = get_batch(dataset, batch_size)
        x_list.extend(x)
        spinal_mask_list.extend(spinal_cord_mask)
        gm_mask_list.extend(gm_mask)
    return torch.stack(x_list, dim=0).cuda(), torch.stack(spinal_mask_list, dim=0).cuda(), torch.stack(gm_mask_list,
                                                                                                       dim=0).cuda()


def train_net(args):
    if args.load != "":
        checkpoint = torch.load(args.load + '/saved_model')
        net_spinal_cord, net_gm = checkpoint['net_spinal_cord'].cuda(), checkpoint['net_gm'].cuda()
    else:
        net_spinal_cord = unet.UNetOurs(1, 1, feature_dim=args.latent_dim).cuda()
        net_gm = unet.UNetOurs(1, 1, feature_dim=args.latent_dim).cuda()
    dataset_list = spinal_cord_challenge.makeDataset(phase='train_nips',
                                                     transform_train=T.ComposedTransform(
                                                         [T.CenterCrop(160),
                                                          # T.Sharpness([0, 30]),
                                                          # T.Blurriness(), T.Noise([0., 0.05]),
                                                          # T.Brightness(),
                                                          # T.Rotation(), T.Scale([0.7, 1.3]),
                                                          T.RandomCrop(144)]))
    target_domain_dataset = dataset_list.pop('site%d' % args.d_t)
    target_domain_dataset.phase = 'infer'
    source_domain_datasets = list(dataset_list.values())
    source_domain_num = len(source_domain_datasets)

    total_sample_num = sum([len(source_domain_dataset) for source_domain_dataset in source_domain_datasets])
    iter_per_epoch = total_sample_num // args.batch_size

    # train_loader_list = [DataLoader(source_domain_dataset, batch_size=args.batch_size, shuffle=True,
    #                                 pin_memory=False) for source_domain_dataset in source_domain_datasets]
    val_loader = DataLoader(target_domain_dataset, batch_size=1, shuffle=False,
                            pin_memory=True)
    writer = SummaryWriter(args.load,
                           comment='ours_no_meta target:%d_lr:%.2e_batch:%d low_rank_w:%.2e kl_w:%.2e latent_dim:%d %s' % (
                               args.d_t, args.lr, args.batch_size, args.low_rank_tradeoff, args.kl_weight,
                               args.latent_dim, args.info))

    logger = utils.logger.Logger(file_path=os.path.join(writer.logdir, 'log.txt'), tensorboard_writer=writer)

    log_dir = writer.logdir
    global_step = 0
    print(args)
    optimizer_spinal_cord = optim.Adam(net_spinal_cord.parameters(), lr=args.lr, weight_decay=1e-8)
    optimizer_gm = optim.Adam(net_gm.parameters(), lr=args.lr, weight_decay=1e-8)
    lr_scheduler_gm = optim.lr_scheduler.StepLR(optimizer=optimizer_gm, step_size=80)
    lr_scheduler_spinal = optim.lr_scheduler.StepLR(optimizer=optimizer_spinal_cord, step_size=80)
    best_DSC = 0

    for epoch in range(args.epochs):
        net_gm.train()
        net_spinal_cord.train()
        for idx in range(iter_per_epoch):
            # source_datasets_backup = source_domain_datasets.copy()
            random.shuffle(source_domain_datasets)
            x_tr, spinal_cord_mask_tr, gm_mask_tr = get_multi_batch(source_domain_datasets[:-1],
                                                                    args.batch_size // source_domain_num)
            x_te, spinal_cord_mask_te, gm_mask_te = get_multi_batch([source_domain_datasets[-1]],
                                                                    args.batch_size // source_domain_num)
            gm_mask = torch.cat([gm_mask_tr, gm_mask_te], dim=0)  # b*1*w*h
            spinal_cord_mask = torch.cat([spinal_cord_mask_tr, spinal_cord_mask_te], dim=0)
            x = torch.cat([x_tr, x_te], dim=0)  # b*1*w*h

            spinal_cord_pred, mu_logvar = net_spinal_cord(x)  # b*1*w*h, b*64*w*h

            spinal_pos_weight = torch.tensor(1.) / torch.mean(spinal_cord_mask.detach()) * args.p_weight1
            if torch.isinf(spinal_pos_weight) or torch.isnan(spinal_pos_weight):
                spinal_pos_weight = torch.tensor(1.).cuda()

            loss_spinal_cord = F.binary_cross_entropy_with_logits(spinal_cord_pred, spinal_cord_mask,
                                                                  pos_weight=spinal_pos_weight)
            loss_kl_spinal = kl_gaussian_loss(mu_logvar[:, 0], mu_logvar[:, 1])

            # loss_spinal = args.kl_weight * loss_kl_spinal + loss_spinal_cord

            feature = mu_logvar[:, 2].permute(0, 2, 3, 1).contiguous().view(-1, net_spinal_cord.feature_dim)
            feature = feature[torch.randperm(len(feature))]
            U, S_spinal, V = torch.svd(feature[0:2000])
            low_rank_loss_spinal = S_spinal[2]

            total_loss_spin = loss_spinal_cord + loss_kl_spinal * args.kl_weight * 2 + low_rank_loss_spinal * args.low_rank_tradeoff 
            optimizer_spinal_cord.zero_grad()
            total_loss_spin.backward()
            optimizer_spinal_cord.step()
            # gm segmentation # # gm segmentation # # gm segmentation # # gm segmentation # # gm segmentation #

            spinal_mask_pred = (torch.sigmoid(spinal_cord_pred) > 0.5).detach().float()  # N*C*W*H
            local_max = (spinal_mask_pred * x).max(dim=2)[0].max(dim=2)[0]
            local_min = ((1 - spinal_mask_pred) * 9999 + spinal_mask_pred * x).min(dim=2)[0].min(dim=2)[0]
            local_min *= (local_min < 9000).float()
            local_max = local_max.view(-1, 1, 1, 1)
            local_min = local_min.view(-1, 1, 1, 1)
            x = torch.clamp((x - local_min) / ((local_max - local_min) + ((local_max - local_min) == 0).float()), 0, 1)
            gm_pred, mu_logvar = net_gm(x)

            gm_pos_weight = torch.sum(spinal_mask_pred) / torch.sum(spinal_mask_pred * gm_mask)
            if torch.isinf(gm_pos_weight) or torch.isnan(gm_pos_weight):
                gm_pos_weight = torch.tensor(1.).cuda()
            loss_gm = F.binary_cross_entropy_with_logits(
                gm_pred * spinal_mask_pred,
                gm_mask,
                pos_weight=gm_pos_weight)

            loss_kl_gm = kl_gaussian_loss(mu_logvar[:, 0], mu_logvar[:, 1])

            feature = mu_logvar[:, 2].permute(0, 2, 3, 1).contiguous().view(-1, net_gm.feature_dim)
            spinal_mask_local = spinal_mask_pred.permute(0, 2, 3, 1).view(-1).nonzero()[:, 0]
            feature = feature[spinal_mask_local][torch.randperm(len(spinal_mask_local))]
            U, S_gm, V = torch.svd(feature[0:min(2000, len(spinal_mask_local))])
            # S_, S_indices = torch.sort(S_gm, dim=1, descending=True)
            low_rank_loss_gm = S_gm[2]

            total_loss_gm = args.kl_weight * loss_kl_gm + loss_gm + low_rank_loss_gm * args.low_rank_tradeoff
            optimizer_gm.zero_grad()
            total_loss_gm.backward()
            optimizer_gm.step()

            logger.collect_iter_info(
                {'gm_total_loss': total_loss_gm, 'spinal_total_loss': total_loss_spin, 'gm_rec_loss': loss_gm,
                 'gm_low_rank_loss': low_rank_loss_gm, 'gm_kl_loss': loss_kl_gm, 'spinal_rec_loss': loss_spinal_cord,
                 'spinal_low_rank_loss': low_rank_loss_spinal, 'spinal_kl_loss': loss_kl_spinal,
                 'rank_spinal': S_spinal, 'rank_gm': S_gm})
            global_step += 1

        logger.log_train_info(epoch=epoch)
        if epoch % 1 == 0 or epoch == args.epochs:
            eval_result = eval.eval_net(net_spinal_cord, net_gm, val_loader, writer=writer, epoch=epoch, logger=logger)
            if eval_result['DSC'] > best_DSC:
                best_DSC = eval_result['DSC']
                torch.save({'net_spinal_cord': net_spinal_cord, 'net_gm': net_gm},
                           os.path.join(writer.logdir, 'best_dsc_model.pth'))
                with open(os.path.join(writer.logdir, 'best_dsc.txt'), 'a') as f:
                    f.write('epoch:%d\n' % epoch)
                    f.write(str(eval_result))
                    f.write('\n')
        lr_scheduler_gm.step()
        lr_scheduler_spinal.step()
        # torch.save({'net_spinal_cord': net_spinal_cord, 'net_gm': net_gm},
        #            os.path.join(log_dir, 'saved_model'))
        writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--target domain', dest='d_t', type=int, default=1, help="target domain(from 1 to 4)")
    parser.add_argument('-k', '--kl_weight', type=float, default=0.01,  # 366
                        help='kl loss tradeoff', dest='kl_weight')
    parser.add_argument('-lk', '--low_rank_tradeoff', type=float, default=0.001,  # 366
                        help='low rank loss tradeoff', dest='low_rank_tradeoff')
    parser.add_argument('-w1', '--weight_spinal', type=float, default=2,  # 366
                        help='extra spinal cord weight', dest='p_weight1')
    parser.add_argument('-i', '--info', type=str, default='', help='comment info', dest='info')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,  # 5e-6,
                        help='Learning rate', dest='lr')

    parser.add_argument('--latent_dim', type=int, default=8,
                        help='latent dim of vae')
    parser.add_argument('-f', '--load', dest='load', type=str,
                        default='',  
                        help='Load model from a .pth file')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=24,
                        help='Number of epochs', dest='batch_size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train_net(args)

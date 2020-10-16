import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_box, get_center
import torch.nn as nn
from utils import metrics
import utils
import os
from dataset import spinal_cord_challenge
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import SynchronousTransforms.transforms as T
import surface_distance
from sklearn import metrics as skmetrics
import numpy as np


def eval_net(net_spinal_cord, net_gm, loader, writer, logger, threshold=0.5, epoch=None):
    """Evaluation without the densecrf with the dice coefficient"""
    net_gm.eval()
    net_spinal_cord.eval()
    resolution = loader.dataset.info_dict['resolution']
    criterion = nn.BCEWithLogitsLoss()
    metric_dict = {'DSC': metrics.DiceSimilarityCoefficient, 'JI': metrics.JaccardIndex,
                   'CC': metrics.ConformityCoefficient,
                   'TPR': metrics.Sensitivity, 'TNR': metrics.Speciﬁcity, 'PPV': metrics.Precision}

    transform_eval = T.ComposedTransform([T.CenterCrop(144)])
    eval_result = {key: 0 for key in metric_dict.keys()}
    eval_result['loss_gm'] = 0
    eval_result['loss_spinal_cord'] = 0
    eval_result['auc'] = 0
    eval_result['avg_surface_distance'] = 0
    with torch.no_grad():
        for data3D, gt_list in loader:
            gt_list = torch.cat([gt.byte() for gt in gt_list], 0)
            x, gt = data3D.transpose(0, 1), gt_list.transpose(0, 1)
            x_max = x.max(dim=2)[0].max(dim=2)[0]
            x_max = (x_max + (x_max == 0).float()).view(-1, 1, 1, 1)
            x = x / x_max
            true_masks = gt.cuda()
            spinal_cord_mask = (torch.mean((true_masks > 0).float(), dim=1) > 0.5).unsqueeze(dim=1).float()
            gm_gt_mask = (torch.mean((true_masks == 1).float(), dim=1) > 0.5).unsqueeze(
                dim=1).float()
            trans_x, trans_cord_mask, trans_gm_mask = [], [], []
            for i in range(x.shape[0]):
                a, b, c = transform_eval(x[i], spinal_cord_mask[i], gm_gt_mask[i])
                trans_x.append(a), trans_cord_mask.append(b), trans_gm_mask.append(c)

            x, spinal_cord_mask, gm_gt_mask = torch.stack(trans_x, dim=0).cuda(), torch.stack(trans_cord_mask,
                                                                                              dim=0).cuda(), torch.stack(
                trans_gm_mask, dim=0).cuda()
            spinal_cord_pred, _ = net_spinal_cord(x)
            loss_spinal_cord = criterion(spinal_cord_pred, spinal_cord_mask)

            spinal_mask_pred = (torch.sigmoid(spinal_cord_pred) > 0.5).detach().float()
            local_max = (spinal_mask_pred * x).max(dim=2)[0].max(dim=2)[0]
            local_min = ((1 - spinal_mask_pred) * 9999 + spinal_mask_pred * x).min(dim=2)[0].min(dim=2)[0]

            local_max = local_max.view(-1, 1, 1, 1)
            local_min = local_min.view(-1, 1, 1, 1)
            local_min *= (local_min < 9000).float()
            x = torch.clamp((x - local_min) / ((local_max - local_min) + ((local_max - local_min) == 0).float()), 0, 1)
            gm_pred, _ = net_gm(x)  # * spinal_mask_pred
            gm_pos_weight = torch.sum(spinal_cord_mask) / torch.sum(spinal_cord_mask * gm_gt_mask)
            if torch.isinf(gm_pos_weight) or torch.isnan(gm_pos_weight):
                gm_pos_weight = torch.tensor(1.).cuda()
            loss_gm = F.binary_cross_entropy_with_logits(gm_pred * spinal_mask_pred, gm_gt_mask,
                                                         pos_weight=gm_pos_weight)
            gm_pred_mask = torch.sigmoid(gm_pred * spinal_mask_pred) > 0.5
            eval_result['loss_gm'] += loss_gm.item()
            eval_result['loss_spinal_cord'] += loss_spinal_cord.item()
            eval_result['auc'] += skmetrics.roc_auc_score(gm_gt_mask.view(-1).cpu().numpy(),
                                                          gm_pred.view(-1).cpu().numpy())
            surface_dis = surface_distance.compute_surface_distances(
                gm_gt_mask.cpu().squeeze().numpy().astype(np.bool),
                gm_pred_mask.cpu().squeeze().numpy().astype(np.bool), spacing_mm=resolution)
            eval_result['avg_surface_distance'] += np.mean(
                surface_distance.compute_average_surface_distance(surface_dis))
            # gm_pred_full_size = (torch.sigmoid(gm_pred) * spinal_mask_pred) > 0.5

            # writer.add_images('raw_data', x, global_step=epoch)
            # writer.add_images('SpinalList_Pred', spinal_mask_pred, global_step=epoch)
            # writer.add_images('SpinalList_GT', spinal_cord_mask, global_step=epoch)
            # writer.add_images('GmMask_Pred', gm_pred_mask, global_step=epoch)
            # writer.add_images('GmMask_GT', gm_gt_mask, global_step=epoch)
            for key, metric_func in metric_dict.items():
                eval_result[key] += metric_func(gm_pred_mask, gm_gt_mask)

        for key, val in eval_result.items():
            eval_result[key] = val / len(loader)
            # writer.add_scalar('eval/' + key, eval_result[key], global_step=epoch)
        info_list = [{'name': key, 'val': value} for key, value in eval_result.items()]
        logger.log_epoch_info(info_list, epoch=epoch)
        return eval_result


if __name__ == '__main__':
    target_domain = 1
    gpu_id = utils.get_available_GPUs(1, 1, 0.5)[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print("GPU_ID:%d" % gpu_id)
    checkpoint = torch.load(
        '/home/hlli/project/yufei/Medical_Segmentation/runs/Mar02_08-33-01_hlli-U2004/saved_model')
    writer = SummaryWriter('/home/hlli/project/yufei/Medical_Segmentation/runs/Mar01_13-54-51_hlli-U2004')
    dataset_list = spinal_cord_challenge.makeDataset(phase='train', specific_domain=['site%d' % target_domain])
    target_domain_dataset = dataset_list.pop('site%d' % target_domain)
    target_domain_dataset.phase = 'infer'
    val_loader = DataLoader(target_domain_dataset, batch_size=1, shuffle=False,
                            pin_memory=True)
    eval_net(checkpoint['net_spinal_cord'].cuda(), checkpoint['net_gm'].cuda(), loader=val_loader,
             writer=writer)

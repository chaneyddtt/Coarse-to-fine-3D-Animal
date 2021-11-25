from __future__ import print_function, absolute_import, division

import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.model_v1 import MeshModel
from util import config
from util.helpers.visualize import Visualizer
from util.loss_utils import kp_l2_loss, Shape_prior, mask_loss
from util.metrics import Metrics
from datasets.stanford import BaseDataset
from util.logger import Logger
from util.meter import AverageMeterSet
from util.misc import save_checkpoint, adjust_learning_rate_exponential
from util.pose_prior import Prior
from util.joint_limits_prior import LimitPrior
import pickle

# Set some global varibles
global_step = 0
best_pck = 0
best_pck_epoch = 0


def main(args):
    global best_pck
    global best_pck_epoch
    global global_step
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print("RESULTS: {0}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set up model
    model = MeshModel(device, args.shape_family_id, args.betas_scale, args.shape_init, render_rgb=args.save_results)
    model = nn.DataParallel(model).to(device)
    # set up datasets
    dataset_train = BaseDataset(args.dataset, param_dir=args.param_dir, is_train=True, use_augmentation=True)
    dataset_eval = BaseDataset(args.dataset, param_dir=args.param_dir, is_train=False, use_augmentation=False)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)
    data_loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(os.path.join(args.output_dir, 'train'))

    # set up criterion
    joint_limit_prior = LimitPrior(device)
    shape_prior = Shape_prior(args.prior_betas, args.shape_family_id, device)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            if args.load_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
            print("=> loaded checkpoint {} (epoch {})".format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(args.output_dir, 'log.txt'))
            logger.log_arguments(args)
            logger.set_names(['Epoch', 'LR', 'PCK', 'IOU'])
        else:
            print("=> no checkpoint found at {}".format(args.resume))
    else:
        logger = Logger(os.path.join(args.output_dir, 'log.txt'))
        logger.log_arguments(args)
        logger.set_names(['Epoch', 'LR', 'PCK', 'IOU'])

    if args.evaluate:
        pck, iou_silh, pck_by_part = run_evaluation(model, dataset_eval, data_loader_eval, device, args)
        print("Evaluate only, PCK: {}, IOU: {}".format(pck, iou_silh))
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.nEpochs):

        model.train()
        tqdm_iterator = tqdm(data_loader_train, desc='Train', total=len(data_loader_train))
        meters = AverageMeterSet()

        for step, batch in enumerate(tqdm_iterator):
            keypoints = batch['keypoints'].to(device)
            seg = batch['seg'].to(device)
            img = batch['img'].to(device)

            pred_codes = model(img)
            scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred = pred_codes
            pred_camera = torch.cat([scale_pred[:, [0]], torch.ones(keypoints.shape[0], 2).cuda() * config.IMG_RES / 2],
                                    dim=1)
            # recover 3D mesh from SMAL parameters
            verts, joints, _, _ = model.module.smal(betas_pred, pose_pred, trans=trans_pred,
                                                    betas_logscale=betas_scale_pred)
            faces = model.module.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)
            labelled_joints_3d = joints[:, config.MODEL_JOINTS]
            # project 3D joints onto 2D space and apply 2D keypoints supervision
            synth_landmarks = model.module.model_renderer.project_points(labelled_joints_3d, pred_camera)
            loss_kpts = args.w_kpts * kp_l2_loss(synth_landmarks, keypoints[:, :, [1, 0, 2]], config.NUM_JOINTS)
            meters.update('loss_kpt', loss_kpts.item())
            loss = loss_kpts

            # apply shape prior constraint, either come from SMAL or unity from WLDO
            if args.w_betas_prior > 0:
                if args.prior_betas == 'smal':
                    s_prior = args.w_betas_prior * shape_prior(betas_pred)
                elif args.prior_betas == 'unity':
                    betas_pred = torch.cat([betas_pred, betas_scale_pred], dim=1)
                    s_prior = args.w_betas_prior * shape_prior(betas_pred)
                else:
                    Exception("Shape prior should come from either smal or unity")
                    s_prior = 0
                meters.update('loss_prior', s_prior.item())
                loss += s_prior

            # apply pose prior constraint, either come from SMAL or unity from WLDO
            if args.w_pose_prior > 0:
                if args.prior_pose == 'smal':
                    pose_prior_path = config.WALKING_PRIOR_FILE
                elif args.prior_pose == 'unity':
                    pose_prior_path = config.UNITY_POSE_PRIOR
                else:
                    Exception('The prior should come from either smal or unity')
                    pose_prior_path = None
                pose_prior = Prior(pose_prior_path, device)
                p_prior = args.w_pose_prior * pose_prior(pose_pred)
                meters.update('pose_prior', p_prior.item())
                loss += p_prior

            meters.update('loss_all', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if step % 20 == 0:
                loss_values = meters.averages()
                for name, meter in loss_values.items():
                    writer.add_scalar(name, meter, global_step)
                writer.flush()

        pck, iou_silh, pck_by_part = run_evaluation(model, dataset_eval, data_loader_eval, device, args)

        print("Epoch: {}, LR: {}, PCK: {}, IOU: {}".format(epoch, lr, pck, iou_silh))
        logger.append([epoch, lr, pck, iou_silh])

        is_best = pck > best_pck
        if pck > best_pck:
            best_pck_epoch = epoch
        best_pck = max(pck, best_pck)
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_pck': best_pck,
                         'optimizer': optimizer.state_dict()},
                          is_best, checkpoint=args.output_dir, filename='checkpoint.pth.tar')
    writer.close()
    logger.close()


def run_evaluation(model, dataset, data_loader, device, args):
    model.eval()
    result_dir = args.output_dir
    batch_size = args.batch_size

    pck = np.zeros((len(dataset)))
    pck_by_part = {group: np.zeros((len(dataset))) for group in config.KEYPOINT_GROUPS}
    acc_sil_2d = np.zeros(len(dataset))

    smal_pose = np.zeros((len(dataset), 105))
    smal_betas = np.zeros((len(dataset), 20))
    smal_camera = np.zeros((len(dataset), 3))
    smal_imgname = []

    tqdm_iterator = tqdm(data_loader, desc='Eval', total=len(data_loader))

    for step, batch in enumerate(tqdm_iterator):
        with torch.no_grad():
            keypoints = batch['keypoints'].to(device)
            keypoints_norm = batch['keypoints_norm'].to(device)
            seg = batch['seg'].to(device)
            has_seg = batch['has_seg']
            img = batch['img'].to(device)
            img_border_mask = batch['img_border_mask'].to(device)
            pred_codes = model(img)

            scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred = pred_codes
            pred_camera = torch.cat([scale_pred[:, [0]], torch.ones(keypoints.shape[0], 2).cuda() * config.IMG_RES / 2],
                                    dim=1)
            verts, joints, _, _ = model.module.smal(betas_pred, pose_pred, trans=trans_pred,
                                                    betas_logscale=betas_scale_pred)
            faces = model.module.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)
            labelled_joints_3d = joints[:, config.MODEL_JOINTS]
            synth_rgb, synth_silhouettes = model.module.model_renderer(verts, faces, pred_camera)
            if args.save_results:
                synth_rgb = torch.clamp(synth_rgb[0], 0.0, 1.0)
            synth_silhouettes = synth_silhouettes.unsqueeze(1)
            synth_landmarks = model.module.model_renderer.project_points(labelled_joints_3d, pred_camera)

            preds = {}
            preds['pose'] = pose_pred
            preds['betas'] = betas_pred
            preds['camera'] = pred_camera
            preds['trans'] = trans_pred

            preds['verts'] = verts
            preds['joints_3d'] = labelled_joints_3d
            preds['faces'] = faces

            preds['acc_PCK'] = Metrics.PCK(synth_landmarks, keypoints_norm, seg, has_seg)
            preds['acc_IOU'] = Metrics.IOU(synth_silhouettes, seg, img_border_mask, mask=has_seg)

            for group, group_kps in config.KEYPOINT_GROUPS.items():
                preds[f'{group}_PCK'] = Metrics.PCK(synth_landmarks, keypoints_norm, seg, has_seg,
                                                    thresh_range=[0.15],
                                                    idxs=group_kps)

            preds['synth_xyz'] = synth_rgb
            preds['synth_silhouettes'] = synth_silhouettes
            preds['synth_landmarks'] = synth_landmarks

            assert not any(k in preds for k in batch.keys())
            preds.update(batch)

        curr_batch_size = preds['synth_landmarks'].shape[0]

        pck[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_PCK'].data.cpu().numpy()
        acc_sil_2d[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_IOU'].data.cpu().numpy()
        smal_pose[step * batch_size:step * batch_size + curr_batch_size] = preds['pose'].data.cpu().numpy()
        smal_betas[step * batch_size:step * batch_size + curr_batch_size, :preds['betas'].shape[1]] = preds['betas'].data.cpu().numpy()
        smal_camera[step * batch_size:step * batch_size + curr_batch_size] = preds['camera'].data.cpu().numpy()

        for part in pck_by_part:
            pck_by_part[part][step * batch_size:step * batch_size + curr_batch_size] = preds[f'{part}_PCK'].data.cpu().numpy()

        # save results as well as visualization
        if args.save_results:
            output_figs = np.transpose(
                Visualizer.generate_output_figures(preds).data.cpu().numpy(),
                (0, 1, 3, 4, 2))

            for img_id in range(len(preds['imgname'])):
                imgname = preds['imgname'][img_id]
                output_fig_list = output_figs[img_id]

                path_parts = imgname.split('/')
                path_suffix = "{0}_{1}".format(path_parts[-2], path_parts[-1])
                img_file = os.path.join(result_dir, path_suffix)
                output_fig = np.hstack(output_fig_list)
                smal_imgname.append(path_suffix)
                # npz_file = "{0}.npz".format(os.path.splitext(img_file)[0])

                cv2.imwrite(img_file, output_fig[:, :, ::-1] * 255.0)
                # np.savez_compressed(npz_file,
                #                     imgname=preds['imgname'][img_id],
                #                     pose=preds['pose'][img_id].data.cpu().numpy(),
                #                     betas=preds['betas'][img_id].data.cpu().numpy(),
                #                     camera=preds['camera'][img_id].data.cpu().numpy(),
                #                     trans=preds['trans'][img_id].data.cpu().numpy(),
                #                     acc_PCK=preds['acc_PCK'][img_id].data.cpu().numpy(),
                #                     # acc_SIL_2D=preds['acc_IOU'][img_id].data.cpu().numpy(),
                #                     **{f'{part}_PCK': preds[f'{part}_PCK'].data.cpu().numpy() for part in pck_by_part}
                #                     )

    return np.nanmean(pck), np.nanmean(acc_sil_2d), pck_by_part


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--output_dir', default='./logs/', type=str)
    parser.add_argument('--nEpochs', default=200, type=int)
    parser.add_argument('--w_kpts', default=10, type=float)
    parser.add_argument('--w_betas_prior', default=1, type=float)
    parser.add_argument('--w_pose_prior', default=1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_works', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--shape_family_id', default=1, type=int)
    parser.add_argument('--dataset', default='stanford', type=str)
    parser.add_argument('--param_dir', default=None, type=str, help='Exported parameter folder to load')
    parser.add_argument('--shape_init', default='smal', help='enable to initiate shape with mean shape')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--prior_betas', default='smal', type=str)
    parser.add_argument('--prior_pose', default='smal', type=str)
    parser.add_argument('--betas_scale', action='store_true')

    args = parser.parse_args()
    main(args)
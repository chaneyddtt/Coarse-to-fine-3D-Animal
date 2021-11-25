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
from model.mesh_graph_hg import MeshGraph_hg, init_pretrained

from util import config
from util.helpers.visualize import Visualizer
from util.loss_utils import kp_l2_loss, Shape_prior, Laplacian
from util.loss_sdf import tversky_loss
from util.metrics import Metrics

from datasets.stanford import BaseDataset

from util.logger import Logger
from util.meter import AverageMeterSet
from util.misc import save_checkpoint
from util.pose_prior import Prior
from util.joint_limits_prior import LimitPrior
from util.utils import print_options

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
    model = MeshGraph_hg(device, args.shape_family_id, args.num_channels, args.num_layers, args.betas_scale,
                      args.shape_init, args.local_feat, num_downsampling=args.num_downsampling,
                      render_rgb=args.save_results)

    model = nn.DataParallel(model).to(device)

    # set up dataset
    dataset_train = BaseDataset(args.dataset, param_dir=args.param_dir, is_train=True, use_augmentation=True)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works)
    dataset_eval = BaseDataset(args.dataset, param_dir=args.param_dir, is_train=False, use_augmentation=False)
    data_loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(os.path.join(args.output_dir, 'train'))

    # set up priors
    joint_limit_prior = LimitPrior(device)
    shape_prior = Shape_prior(args.prior_betas, args.shape_family_id, device)
    tversky = tversky_loss(args.alpha, args.beta)

    # read the adjacency matrix, which will used in the Laplacian regularizer
    data = np.load('./data/mesh_down_sampling_4.npz', encoding='latin1', allow_pickle=True)
    adjmat = data['A'][0]
    laplacianloss = Laplacian(adjmat, device)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            if args.load_optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
            print("=> loaded checkpoint {} (epoch {})".format(args.resume, checkpoint['epoch']))
            # logger = Logger(os.path.join(args.output_dir, 'log.txt'), resume=True)
            logger = Logger(os.path.join(args.output_dir, 'log.txt'))
            logger.set_names(['Epoch', 'LR', 'PCK', 'IOU', 'PCK_re', 'IOU_re'])
        else:
            print("=> no checkpoint found at {}".format(args.resume))
    else:
        logger = Logger(os.path.join(args.output_dir, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'PCK', 'IOU','PCK_re', 'IOU_re'])

    if args.freezecoarse:
        for p in model.module.meshnet.parameters():
            p.requires_grad = False
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint {}".format(args.pretrained))
            checkpoint_pre = torch.load(args.pretrained)
            init_pretrained(model, checkpoint_pre)
            print("=> loaded checkpoint {} (epoch {})".format(args.pretrained, checkpoint_pre['epoch']))
            # logger = Logger(os.path.join(args.output_dir, 'log.txt'), resume=True)
            logger = Logger(os.path.join(args.output_dir, 'log.txt'))
            logger.set_names(['Epoch', 'LR', 'PCK', 'IOU', 'PCK_re', 'IOU_re'])

    print_options(args)
    if args.evaluate:
        pck, iou_silh, pck_by_part, pck_re, iou_re = run_evaluation(model, dataset_eval, data_loader_eval, device, args)
        print("Evaluate only, PCK: {:6.4f}, IOU: {:6.4f}, PCK_re: {:6.4f}, IOU_re: {:6.4f}"
              .format(pck, iou_silh, pck_re, iou_re))
        return

    lr = args.lr
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, [160, 190], 0.5,
    # )
    for epoch in range(args.start_epoch, args.nEpochs):
        # lr_scheduler.step()
        # lr = lr_scheduler.get_last_lr()[0]
        model.train()
        tqdm_iterator = tqdm(data_loader_train, desc='Train', total=len(data_loader_train))
        meters = AverageMeterSet()
        for step, batch in enumerate(tqdm_iterator):
            keypoints = batch['keypoints'].to(device)
            keypoints_norm = batch['keypoints_norm'].to(device)
            seg = batch['seg'].to(device)
            img = batch['img'].to(device)

            verts, joints, shape, pred_codes = model(img)
            scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred = pred_codes
            pred_camera = torch.cat([scale_pred[:, [0]], torch.ones(keypoints.shape[0], 2).cuda() * config.IMG_RES / 2],
                                    dim=1)
            faces = model.module.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)
            labelled_joints_3d = joints[:, config.MODEL_JOINTS]

            # project 3D joints onto 2D space and apply 2D keypoints supervision
            synth_landmarks = model.module.model_renderer.project_points(labelled_joints_3d, pred_camera)
            loss_kpts = args.w_kpts * kp_l2_loss(synth_landmarks, keypoints[:, :, [1, 0, 2]], config.NUM_JOINTS)
            meters.update('loss_kpt', loss_kpts.item())
            loss = loss_kpts

            # use tversky for silhouette loss
            if args.w_dice>0:
                synth_rgb, synth_silhouettes = model.module.model_renderer(verts, faces, pred_camera)
                synth_silhouettes = synth_silhouettes.unsqueeze(1)
                loss_dice = args.w_dice * tversky(synth_silhouettes, seg)
                meters.update('loss_dice', loss_dice.item())
                loss += loss_dice

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

            # apply pose limit constraint
            if args.w_pose_limit_prior > 0:
                pose_limit_loss = args.w_pose_limit_prior * joint_limit_prior(pose_pred)
                meters.update('pose_limit', pose_limit_loss.item())
                loss += pose_limit_loss

            # get refined meshes by adding del_v to the coarse mesh from SMAL
            verts_refine, joints_refine, _, _ = model.module.smal(betas_pred, pose_pred, trans=trans_pred,
                                                                  del_v=shape,
                                                                  betas_logscale=betas_scale_pred)
            # apply 2D keypoint and silhouette supervision
            labelled_joints_3d_refine = joints_refine[:, config.MODEL_JOINTS]
            synth_landmarks_refine = model.module.model_renderer.project_points(labelled_joints_3d_refine,
                                                                                pred_camera)
            loss_kpts_refine = args.w_kpts_refine * kp_l2_loss(synth_landmarks_refine, keypoints[:, :, [1, 0, 2]],
                                                        config.NUM_JOINTS)

            meters.update('loss_kpt_refine', loss_kpts_refine.item())
            loss += loss_kpts_refine
            if args.w_dice_refine> 0:
                _, synth_silhouettes_refine = model.module.model_renderer(verts_refine, faces, pred_camera)
                synth_silhouettes_refine = synth_silhouettes_refine.unsqueeze(1)
                loss_dice_refine = args.w_dice_refine * tversky(synth_silhouettes_refine, seg)
                meters.update('loss_dice_refine', loss_dice_refine.item())
                loss += loss_dice_refine

            # apply Laplacian constraint to prevent large deformation predictions
            if args.w_arap > 0:
                verts_clone = verts.detach().clone()
                loss_arap, loss_smooth = laplacianloss(verts_refine, verts_clone)
                loss_arap = args.w_arap * loss_arap
                meters.update('loss_arap', loss_arap.item())
                loss += loss_arap

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

        pck, iou_silh, pck_by_part, pck_re, iou_re = run_evaluation(model, dataset_eval, data_loader_eval, device, args)

        print("Epoch: {:3d}, LR: {:6.5f}, PCK: {:6.4f}, IOU: {:6.4f}, PCK_re: {:6.4f}, IOU_re: {:6.4f}"
              .format(epoch, lr, pck, iou_silh, pck_re, iou_re))
        logger.append([epoch, lr, pck, iou_silh, pck_re, iou_re])

        is_best = pck_re > best_pck
        if pck_re > best_pck:
            best_pck_epoch = epoch
        best_pck = max(pck_re, best_pck)
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

    pck_re = np.zeros((len(dataset)))
    acc_sil_2d_re = np.zeros(len(dataset))

    smal_pose = np.zeros((len(dataset), 105))
    smal_betas = np.zeros((len(dataset), 20))
    smal_camera = np.zeros((len(dataset), 3))
    smal_imgname = []

    tqdm_iterator = tqdm(data_loader, desc='Eval', total=len(data_loader))

    for step, batch in enumerate(tqdm_iterator):
        with torch.no_grad():
            preds = {}
            keypoints = batch['keypoints'].to(device)
            keypoints_norm = batch['keypoints_norm'].to(device)
            seg = batch['seg'].to(device)
            has_seg = batch['has_seg']
            img = batch['img'].to(device)
            img_border_mask = batch['img_border_mask'].to(device)
            # get coarse meshes and project onto 2D
            verts, joints, shape, pred_codes = model(img)
            scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred = pred_codes
            pred_camera = torch.cat([scale_pred[:, [0]], torch.ones(keypoints.shape[0], 2).cuda() * config.IMG_RES / 2],
                                    dim=1)
            faces = model.module.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)
            labelled_joints_3d = joints[:, config.MODEL_JOINTS]

            synth_rgb, synth_silhouettes = model.module.model_renderer(verts, faces, pred_camera)
            synth_silhouettes = synth_silhouettes.unsqueeze(1)
            synth_landmarks = model.module.model_renderer.project_points(labelled_joints_3d, pred_camera)

            # get refined meshes by adding del_v to coarse estimations
            verts_refine, joints_refine, _, _ = model.module.smal(betas_pred, pose_pred, trans=trans_pred,
                                                                  del_v=shape,
                                                                  betas_logscale=betas_scale_pred)
            labelled_joints_3d_refine = joints_refine[:, config.MODEL_JOINTS]
            # project refined 3D meshes onto 2D
            synth_rgb_refine, synth_silhouettes_refine = model.module.model_renderer(verts_refine, faces, pred_camera)
            synth_silhouettes_refine = synth_silhouettes_refine.unsqueeze(1)
            synth_landmarks_refine = model.module.model_renderer.project_points(labelled_joints_3d_refine,
                                                                                pred_camera)

            if args.save_results:
                synth_rgb = torch.clamp(synth_rgb[0], 0.0, 1.0)
                synth_rgb_refine = torch.clamp(synth_rgb_refine[0], 0.0, 1.0)

            preds['pose'] = pose_pred
            preds['betas'] = betas_pred
            preds['camera'] = pred_camera
            preds['trans'] = trans_pred

            preds['verts'] = verts
            preds['joints_3d'] = labelled_joints_3d
            preds['faces'] = faces

            preds['acc_PCK'] = Metrics.PCK(synth_landmarks, keypoints_norm, seg, has_seg)
            preds['acc_IOU'] = Metrics.IOU(synth_silhouettes, seg, img_border_mask, mask=has_seg)

            preds['acc_PCK_re'] = Metrics.PCK(synth_landmarks_refine, keypoints_norm, seg, has_seg)
            preds['acc_IOU_re'] = Metrics.IOU(synth_silhouettes_refine, seg, img_border_mask, mask=has_seg)

            for group, group_kps in config.KEYPOINT_GROUPS.items():
                preds[f'{group}_PCK'] = Metrics.PCK(synth_landmarks, keypoints_norm, seg, has_seg,
                                                    thresh_range=[0.15],
                                                    idxs=group_kps)

            preds['synth_xyz'] = synth_rgb
            preds['synth_silhouettes'] = synth_silhouettes
            preds['synth_landmarks'] = synth_landmarks
            preds['synth_xyz_re'] = synth_rgb_refine
            preds['synth_landmarks_re'] = synth_landmarks_refine
            preds['synth_silhouettes_re'] = synth_silhouettes_refine

            assert not any(k in preds for k in batch.keys())
            preds.update(batch)

        curr_batch_size = preds['synth_landmarks'].shape[0]

        pck[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_PCK'].data.cpu().numpy()
        acc_sil_2d[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_IOU'].data.cpu().numpy()
        smal_pose[step * batch_size:step * batch_size + curr_batch_size] = preds['pose'].data.cpu().numpy()
        smal_betas[step * batch_size:step * batch_size + curr_batch_size, :preds['betas'].shape[1]] = preds['betas'].data.cpu().numpy()
        smal_camera[step * batch_size:step * batch_size + curr_batch_size] = preds['camera'].data.cpu().numpy()

        pck_re[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_PCK_re'].data.cpu().numpy()
        acc_sil_2d_re[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_IOU_re'].data.cpu().numpy()
        for part in pck_by_part:
            pck_by_part[part][step * batch_size:step * batch_size + curr_batch_size] = preds[f'{part}_PCK'].data.cpu().numpy()

        if args.save_results:
            output_figs = np.transpose(
                Visualizer.generate_output_figures(preds, vis_refine=True).data.cpu().numpy(),
                (0, 1, 3, 4, 2))

            for img_id in range(len(preds['imgname'])):
                imgname = preds['imgname'][img_id]
                output_fig_list = output_figs[img_id]

                path_parts = imgname.split('/')
                path_suffix = "{0}_{1}".format(path_parts[-2], path_parts[-1])
                img_file = os.path.join(result_dir, path_suffix)
                output_fig = np.hstack(output_fig_list)
                smal_imgname.append(path_suffix)
                npz_file = "{0}.npz".format(os.path.splitext(img_file)[0])

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

    return np.nanmean(pck), np.nanmean(acc_sil_2d), pck_by_part, np.nanmean(pck_re), np.nanmean(acc_sil_2d_re)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--output_dir', default='./logs/', type=str)
    parser.add_argument('--nEpochs', default=250, type=int)

    parser.add_argument('--w_kpts', default=10, type=float)
    parser.add_argument('--w_betas_prior', default=1, type=float)
    parser.add_argument('--w_pose_prior', default=1, type=float)
    parser.add_argument('--w_pose_limit_prior', default=0, type=float)
    parser.add_argument('--w_kpts_refine', default=1, type=float)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_works', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--dataset', default='stanford', type=str)
    parser.add_argument('--shape_family_id', default=1, type=int)
    parser.add_argument('--param_dir', default=None, type=str, help='Exported parameter folder to load')

    parser.add_argument('--shape_init', default='smal', help='enable to initiate shape with mean shape')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--prior_betas', default='smal', type=str)
    parser.add_argument('--prior_pose', default='smal', type=str)
    parser.add_argument('--betas_scale', action='store_true')

    parser.add_argument('--num_channels', type=int, default=256, help='Number of channels in Graph Residual layers')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of residuals blocks in the Graph CNN')
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--local_feat', action='store_true')

    parser.add_argument('--num_downsampling', default=1, type=int)
    parser.add_argument('--freezecoarse', action='store_true')

    parser.add_argument('--w_arap', default=0, type=float)
    parser.add_argument('--w_dice', default=0, type=float)
    parser.add_argument('--w_dice_refine', default=0, type=float)
    parser.add_argument('--alpha', default=0.6, type=float)
    parser.add_argument('--beta', default=0.4, type=float)

    args = parser.parse_args()
    main(args)
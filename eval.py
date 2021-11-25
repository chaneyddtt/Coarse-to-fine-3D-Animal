from __future__ import print_function, absolute_import, division

import os
import sys
import time
import numpy as np
from tqdm import tqdm
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from model.mesh_graph_hg import MeshGraph_hg
from util import config
from util.helpers.visualize import Visualizer
from util.metrics import Metrics
from datasets.stanford import BaseDataset
from scipy.spatial.transform import Rotation as R

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # set model
    model = MeshGraph_hg(device, args.shape_family_id, args.num_channels, args.num_layers, args.betas_scale,
                      args.shape_init, args.local_feat, num_downsampling=args.num_downsampling,
                      render_rgb=args.save_results)
    model = nn.DataParallel(model).to(device)
    # set data
    print("Evaluate on {} dataset".format(args.dataset))
    dataset_eval = BaseDataset(args.dataset, param_dir=args.param_dir, is_train=False, use_augmentation=False)
    data_loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if os.path.isfile(args.resume):
        print("=> loading checkpoint {}".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        if args.load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        print("=> loaded checkpoint {} (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("No checkpoint found")

    pck, iou_silh, pck_by_part, pck_re, iou_re = run_evaluation(model, dataset_eval, data_loader_eval, device, args)
    print("Evaluate only, PCK: {:6.4f}, IOU: {:6.4f}, PCK_re: {:6.4f}, IOU_re: {:6.4f}"
          .format(pck, iou_silh, pck_re, iou_re))
    return


def run_evaluation(model, dataset, data_loader, device, args):

    model.eval()
    result_dir = args.output_dir
    batch_size = args.batch_size

    pck = np.zeros((len(dataset)))
    pck_by_part = {group: np.zeros((len(dataset))) for group in config.KEYPOINT_GROUPS}
    pck_by_part_re = {group: np.zeros((len(dataset))) for group in config.KEYPOINT_GROUPS}
    acc_sil_2d = np.zeros(len(dataset))

    pck_re = np.zeros((len(dataset)))
    acc_sil_2d_re = np.zeros(len(dataset))

    smal_pose = np.zeros((len(dataset), 105))
    smal_betas = np.zeros((len(dataset), 20))
    smal_camera = np.zeros((len(dataset), 3))
    smal_imgname = []
    # rotate estimated mesh to visualize in an alternative view
    rot_matrix = torch.from_numpy(R.from_euler('y', -90, degrees=True).as_dcm()).float().to(device)
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
            verts, joints, shape, pred_codes = model(img)
            scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred = pred_codes
            pred_camera = torch.cat([scale_pred[:, [0]], torch.ones(keypoints.shape[0], 2).cuda() * config.IMG_RES / 2],
                                    dim=1)
            faces = model.module.smal.faces.unsqueeze(0).expand(verts.shape[0], 7774, 3)
            labelled_joints_3d = joints[:, config.MODEL_JOINTS]
            synth_rgb, synth_silhouettes = model.module.model_renderer(verts, faces, pred_camera)
            synth_silhouettes = synth_silhouettes.unsqueeze(1)
            synth_landmarks = model.module.model_renderer.project_points(labelled_joints_3d, pred_camera)

            verts_refine, joints_refine, _, _ = model.module.smal(betas_pred, pose_pred, trans=trans_pred,
                                                                  del_v=shape,
                                                                  betas_logscale=betas_scale_pred)
            labelled_joints_3d_refine = joints_refine[:, config.MODEL_JOINTS]
            synth_rgb_refine, synth_silhouettes_refine = model.module.model_renderer(verts_refine, faces, pred_camera)
            synth_silhouettes_refine = synth_silhouettes_refine.unsqueeze(1)
            synth_landmarks_refine = model.module.model_renderer.project_points(labelled_joints_3d_refine,
                                                                                pred_camera)

            if args.save_results:
                synth_rgb = torch.clamp(synth_rgb[0], 0.0, 1.0)
                synth_rgb_refine = torch.clamp(synth_rgb_refine[0], 0.0, 1.0)
                # visualize in another view
                verts_refine_cano = verts_refine - torch.mean(verts_refine, dim=1, keepdim=True)
                verts_refine_cano = (rot_matrix @ verts_refine_cano.unsqueeze(-1)).squeeze(-1)
                # increase the depth such that the rendered the shapes are in within the image
                verts_refine_cano[:, :, 2] = verts_refine_cano[:, :, 2] + 15
                synth_rgb_refine_cano, _ = model.module.model_renderer(verts_refine_cano, faces,
                                                                                         pred_camera)
                synth_rgb_refine_cano = torch.clamp(synth_rgb_refine_cano[0], 0.0, 1.0)
                preds['synth_xyz_re_cano'] = synth_rgb_refine_cano

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
                preds[f'{group}_PCK_RE'] = Metrics.PCK(synth_landmarks_refine, keypoints_norm, seg, has_seg,
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
        # compute accuracy for coarse stage
        pck[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_PCK'].data.cpu().numpy()
        acc_sil_2d[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_IOU'].data.cpu().numpy()
        smal_pose[step * batch_size:step * batch_size + curr_batch_size] = preds['pose'].data.cpu().numpy()
        smal_betas[step * batch_size:step * batch_size + curr_batch_size, :preds['betas'].shape[1]] = preds['betas'].data.cpu().numpy()
        smal_camera[step * batch_size:step * batch_size + curr_batch_size] = preds['camera'].data.cpu().numpy()
        # compute accuracy for refinement stage
        pck_re[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_PCK_re'].data.cpu().numpy()
        acc_sil_2d_re[step * batch_size:step * batch_size + curr_batch_size] = preds['acc_IOU_re'].data.cpu().numpy()
        for part in pck_by_part:
            pck_by_part[part][step * batch_size:step * batch_size + curr_batch_size] = preds[f'{part}_PCK'].data.cpu().numpy()
            pck_by_part_re[part][step * batch_size:step * batch_size + curr_batch_size] = preds[
                f'{part}_PCK_RE'].data.cpu().numpy()

        if args.save_results:
            output_figs = np.transpose(
                Visualizer.generate_output_figures_v2(preds, vis_refine=True).data.cpu().numpy(),
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
    report = f"""*** Final Results ***

    SIL IOU 2D: {np.nanmean(acc_sil_2d):.5f}
    PCK 2D: {np.nanmean(pck):.5f}

    SIL IOU 2D REFINE: {np.nanmean(acc_sil_2d_re):.5f}
    PCK 2D REFINE: {np.nanmean(pck_re):.5f}"""

    for part in pck_by_part:
        report += f'\n   {part} PCK 2D: {np.nanmean(pck_by_part[part]):.5f}'

    for part in pck_by_part:
        report += f'\n   {part} PCK 2D RE: {np.nanmean(pck_by_part_re[part]):.5f}'
    print(report)

    # save report to file
    with open(os.path.join(result_dir, '{}_report.txt'.format(args.dataset)), 'w') as outfile:
        print(report, file=outfile)
    return np.nanmean(pck), np.nanmean(acc_sil_2d), pck_by_part, np.nanmean(pck_re), np.nanmean(acc_sil_2d_re)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--output_dir', default='./logs/', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_works', default=4, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--shape_family_id', default=1, type=int)
    parser.add_argument('--dataset', default='stanford', type=str)
    parser.add_argument('--param_dir', default=None, type=str, help='Exported parameter folder to load')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--prior_betas', default='smal', type=str)
    parser.add_argument('--prior_pose', default='smal', type=str)
    parser.add_argument('--betas_scale', action='store_true')
    parser.add_argument('--num_channels', type=int, default=256, help='Number of channels in Graph Residual layers')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of residuals blocks in the Graph CNN')
    parser.add_argument('--local_feat', action='store_true')
    parser.add_argument('--shape_init', default='smal', help='enable to initiate shape with mean shape')
    parser.add_argument('--num_downsampling', default=1, type=int)

    args = parser.parse_args()
    main(args)
"""
Mesh net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch
from smal.smal_torch import SMAL
from util import config
from model.graph_hg import GraphCNN_hg
from model.smal_mesh_net_img import MeshNet_img
from util.nmr import NeuralRenderer
from smal.mesh import Mesh
import cv2
# ------------- Modules ------------#
# ----------------------------------#

unity_shape_prior = np.load('data/priors/unity_betas.npz')


class MeshGraph_hg(nn.Module):
    def __init__(self, device, shape_family_id, number_channels, num_layers, betas_scale=False, shape_init=None
                 , local_feat=False, num_downsampling=0, render_rgb=False):
        '''

        Args:
            device: specify device for training
            shape_family_id: specify animal category id
            number_channels: specify number of channels for GCN
            betas_scale: whether predict additional shape parameters proposed by WLDO
            shape_init: whether initiate the bias weights for the coarse stage as mean shape
            local_feat: whether use local feature for refinement step
            num_downsampling: number of donwsamplings before input to GCN.
                                We downsample the original mesh once before going through GCN to save memory
            render_rgb: wehther render the 3D mesh onto 2D to get RGB image. Only set to true when generating
            visualization to save inference time.
        '''
        super(MeshGraph_hg, self).__init__()

        self.model_renderer = NeuralRenderer(config.IMG_RES, proj_type=config.PROJECTION,
                                             norm_f0=config.NORM_F0,
                                             norm_f=config.NORM_F,
                                             norm_z=config.NORM_Z, render_rgb=render_rgb, device=device)
        self.model_renderer.directional_light_only()
        self.smal = SMAL(device, shape_family_id=shape_family_id)
        self.local_feat = local_feat
        if shape_init == 'smal':
            print("Initiate shape with smal prior")
            shape_init = self.smal.shape_cluster_means
        elif shape_init == 'unity':
            print("Initiate shape with unity prior ")
            shape_init = unity_shape_prior['mean'][:-1]
            shape_init = torch.from_numpy(shape_init).float().to(device)
        else:
            print("No initialization for shape")
            shape_init = None

        input_size = [config.IMG_RES, config.IMG_RES]
        self.meshnet = MeshNet_img(input_size, betas_scale=betas_scale, norm_f0=config.NORM_F0, nz_feat=config.NZ_FEAT,
                                   shape_init=shape_init, return_feat=True)
        self.mesh = Mesh(self.smal, num_downsampling=num_downsampling, filename='./data/mesh_down_sampling_4.npz',
                         device=device)
        self.graphnet = GraphCNN_hg(self.mesh, num_channels=number_channels,
                                local_feat=local_feat, num_downsample=num_downsampling).to(device)

    def forward(self, img):
        pred_codes, enc_feat, feat_multiscale = self.meshnet(img)
        scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred = pred_codes
        pred_camera = torch.cat([scale_pred[:, [0]], torch.ones(scale_pred.shape[0], 2).cuda() * config.IMG_RES / 2],
                                dim=1)
        verts, joints, _, _ = self.smal(betas_pred, pose_pred, trans=trans_pred,
                                        betas_logscale=betas_scale_pred)
        enc_feat_copy = enc_feat.detach().clone()
        verts_d = self.mesh.downsample(verts)
        verts_copy = verts_d.detach().clone()
        if self.local_feat:
            feat_multiscale_copy = feat_multiscale.detach().clone()
            points_img = self.model_renderer.project_points(verts_copy, pred_camera, normalize_kpts=True)
            # used normalized image coordinate for the requirement of torch.nn.functional.grid_sample
            verts_re = self.graphnet(verts_d, enc_feat_copy, feat_multiscale_copy, points_img.unsqueeze(1))
        else:
            verts_re = self.graphnet(verts_d, enc_feat_copy)
        verts_re = self.mesh.upsample(verts_re.transpose(1, 2))
        return verts, joints, verts_re, pred_codes


def init_pretrained(model, checkpoint):
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Init MeshNet")


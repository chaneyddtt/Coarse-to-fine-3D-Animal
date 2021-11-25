"""
Mesh net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
from util import net_blocks as nb



# ------------- Modules ------------#
# ----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        # if opts.use_resnet50:
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # else:
        #     self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks
        # if self.opts.use_double_input:
        #     self.fc = nb.fc_stack(512*16*8, 512*8*8, 2)

    def forward(self, x, y=None):
        # if self.opts.use_double_input and y is not None:
        #     x = torch.cat([x, y], 2)
        img_feat_multiscale = [] # collect multi-scale features for local feature retrieve
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
            img_feat_multiscale.append(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
            img_feat_multiscale.append(torch.nn.Upsample(size=(56, 56), mode='bilinear')(x))
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
            img_feat_multiscale.append(torch.nn.Upsample(size=(56, 56), mode='bilinear')(x))
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
            img_feat_multiscale.append(torch.nn.Upsample(size=(56, 56), mode='bilinear')(x))
        # if self.opts.use_double_input and y is not None:
        #     x = x.view(x.size(0), -1)
        #     x = self.fc.forward(x)
        #     x = x.view(x.size(0), 512, 8, 8)

        return x, torch.cat(img_feat_multiscale, dim=1)


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, channels_per_group=16, n_blocks=4, nz_feat=100, bott_size=256):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        num_norm_groups = bott_size // channels_per_group
        # if opts.use_resnet50:
        self.enc_conv1 = nb.conv2d('group', 2048, bott_size, stride=2, kernel_size=4, num_groups=num_norm_groups)
        # else:
        #     self.enc_conv1 = nb.conv2d('group', 512, bott_size, stride=2, kernel_size=4, num_groups=num_norm_groups)

        nc_input = bott_size * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2, 'batch')
        self.nenc_feat = nc_input

        nb.net_init(self.enc_conv1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, img, fg_img):
        resnet_feat, feat_multiscale = self.resnet_conv.forward(img, fg_img) # multi-scale feature is used to extract local feature for refinement
        out_enc_conv1 = self.enc_conv1(resnet_feat) # feature for predicting SMAL parameters
        out_resnet = self.avgpool(resnet_feat)   # add an pooling layer to get global feature for mesh refinement
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)
        return feat, out_enc_conv1, out_resnet, feat_multiscale


class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts, left_idx, right_idx, shapedirs, use_delta_v=False, use_sym_idx=False,
                 use_smal_betas=False, n_shape_feat=40):
        super(ShapePredictor, self).__init__()
        self.use_delta_v = use_delta_v
        self.use_sym_idx = use_sym_idx
        self.use_smal_betas = use_smal_betas
        self.ref_delta_v = torch.Tensor(np.zeros((num_verts, 3))).cuda()

    def forward(self, feat):
        if self.use_sym_idx:
            batch_size = feat.shape[0]
            delta_v = torch.Tensor(np.zeros((batch_size, self.num_verts, 3))).cuda()
            feat = self.fc(feat)
            self.shape_f = feat

            half_delta_v = self.pred_layer.forward(feat)
            half_delta_v = half_delta_v.view(half_delta_v.size(0), -1, 3)
            delta_v[:, self.left_idx, :] = half_delta_v
            half_delta_v[:, :, 1] = -1. * half_delta_v[:, :, 1]
            delta_v[:, self.right_idx, :] = half_delta_v
        else:
            delta_v = self.pred_layer.forward(feat)
            # Make it B x num_verts x 3
            delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v


class PosePredictor(nn.Module):
    """
    """

    def __init__(self, nz_feat, num_joints=35):
        super(PosePredictor, self).__init__()
        self.pose_var = 1.0
        self.num_joints = num_joints
        self.pred_layer = nn.Linear(nz_feat, num_joints * 3)
        # bjb_edit
        self.pred_layer.weight.data.normal_(0, 1e-4)
        self.pred_layer.bias.data.normal_(0, 1e-4)

    def forward(self, feat):
        pose = self.pose_var * self.pred_layer.forward(feat)

        # Add this to have zero to correspond to frontal facing
        # edit by lic, frontal facing and upright
        pose[:, 0] += -1.20919958
        pose[:, 1] += 1.20919958
        pose[:, 2] += 1.20919958
        return pose


class BetaScalePredictor(nn.Module):
    def __init__(self, nz_feat, nenc_feat, num_beta_scale=6, model_mean=None):
        super(BetaScalePredictor, self).__init__()
        self.model_mean = model_mean
        self.pred_layer = nn.Linear(nenc_feat, num_beta_scale)
        # bjb_edit
        self.pred_layer.weight.data.normal_(0, 1e-4)
        if model_mean is not None:
            self.pred_layer.bias.data = model_mean + torch.randn_like(model_mean) * 1e-4
        else:
            self.pred_layer.bias.data.normal_(0, 1e-4)

    def forward(self, feat, enc_feat):
        betas = self.pred_layer.forward(enc_feat)

        return betas


class BetasPredictor(nn.Module):
    def __init__(self, nz_feat, nenc_feat, num_betas=20, model_mean=None):
        super(BetasPredictor, self).__init__()
        self.model_mean = model_mean
        self.pred_layer = nn.Linear(nenc_feat, num_betas)
        # bjb_edit
        self.pred_layer.weight.data.normal_(0, 1e-4)
        if model_mean is not None:
            self.pred_layer.bias.data = model_mean + torch.randn_like(model_mean) * 1e-4
        else:
            self.pred_layer.bias.data.normal_(0, 1e-4)

    def forward(self, feat, enc_feat):
        betas = self.pred_layer.forward(enc_feat)
        return betas


class ScalePredictor(nn.Module):
    '''
    In case of perspective projection scale is focal length
    '''

    def __init__(self, nz, norm_f0, use_camera=True, scale_bias=1):
        super(ScalePredictor, self).__init__()
        self.use_camera = use_camera
        self.norm_f0 = norm_f0
        if self.use_camera:
            self.pred_layer = nn.Linear(nz, scale_bias)
        # else:
        #     scale = np.zeros((opts.batch_size,1))
        #     scale[:,0] = 0.
        #     self.ref_camera = torch.Tensor(scale).cuda()

    def forward(self, feat):
        if not self.use_camera:
            scale = np.zeros((feat.shape[0], 1))
            scale[:, 0] = 0.
            return torch.Tensor(scale).cuda()
        if self.norm_f0 != 0:
            off = 0.
        else:
            off = 1.
        scale = self.pred_layer.forward(feat) + off
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, projection_type, fix_trans=False):
        super(TransPredictor, self).__init__()
        self.fix_trans = fix_trans
        if projection_type == 'orth':
            self.pred_layer = nn.Linear(nz, 2)
        elif projection_type == 'perspective':
            self.pred_layer_xy = nn.Linear(nz, 2)
            self.pred_layer_z = nn.Linear(nz, 1)
            self.pred_layer_xy.weight.data.normal_(0, 0.0001)
            self.pred_layer_xy.bias.data.normal_(0, 0.0001)
            self.pred_layer_z.weight.data.normal_(0, 0.0001)
            self.pred_layer_z.bias.data.normal_(0, 0.0001)
        else:
            print('Unknown projection type')

    def forward(self, feat):
        trans = torch.Tensor(np.zeros((feat.shape[0], 3))).cuda()
        f = torch.Tensor(np.zeros((feat.shape[0], 1))).cuda()
        feat_xy = feat
        feat_z = feat

        trans[:, :2] = self.pred_layer_xy(feat_xy)
        trans[:, 2] = 1 + self.pred_layer_z(feat_z)[:, 0]

        if self.fix_trans:
            trans[:, 2] = 1.

        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class CodePredictor(nn.Module):
    def __init__(
            self, norm_f0, nz_feat=100, nenc_feat=2048,
            use_smal_betas=True,
            num_betas=27,  # bjb_edit
            use_camera=True, scale_bias=1,
            fix_trans=False, betas_scale=False,
            use_smal_pose=True,
            shape_init=None):

        super(CodePredictor, self).__init__()
        self.use_smal_pose = use_smal_pose
        self.use_smal_betas = use_smal_betas
        self.use_camera = use_camera
        self.betas_scale = betas_scale
        self.scale_predictor = ScalePredictor(
            nz_feat, norm_f0, use_camera=use_camera, scale_bias=scale_bias)
        self.trans_predictor = TransPredictor(
            nz_feat, 'perspective', fix_trans=fix_trans)

        if self.use_smal_pose:
            self.pose_predictor = PosePredictor(nz_feat)

        if self.use_smal_betas:
            scale_init = None
            shape_betas_init = None
            if shape_init is not None:
                shape_betas_init = shape_init[:20]
                if shape_init.shape[0] == 26:
                    scale_init = shape_init[20:]

            self.betas_predictor = BetasPredictor(
                nz_feat, nenc_feat, num_betas=20, model_mean=shape_betas_init)
            if self.betas_scale:
                self.betas_scale_predictor = BetaScalePredictor(
                    nz_feat, nenc_feat, num_beta_scale=6, model_mean=scale_init)

    def forward(self, feat, enc_feat):
        if self.use_camera:
            scale_pred = self.scale_predictor.forward(feat)
        else:
            scale_pred = self.scale_predictor.ref_camera

        trans_pred = self.trans_predictor.forward(feat)

        if self.use_smal_pose:
            pose_pred = self.pose_predictor.forward(feat)
        else:
            pose_pred = None

        if self.use_smal_betas:
            betas_pred = self.betas_predictor.forward(feat, enc_feat)
            if self.betas_scale:
                betas_scale_pred = self.betas_scale_predictor.forward(feat, enc_feat)[:,
                                   :6]  # choose first 6 for backward compat
            else:
                betas_scale_pred = None
        else:
            betas_pred = None
            betas_scale_pred = None

        return scale_pred, trans_pred, pose_pred, betas_pred, betas_scale_pred


# ------------ Mesh Net ------------#
# ----------------------------------#
class MeshNet_img(nn.Module):
    def __init__(self,
                 input_shape, betas_scale=False,
                 norm_f0=2700., nz_feat=100,
                 shape_init=None, return_feat=False):
        # Input shape is H x W of the image.
        super(MeshNet_img, self).__init__()

        self.bottleneck_size = 2048
        self.channels_per_group = 16
        self.shape_init = shape_init

        self.encoder = Encoder(
            input_shape,
            channels_per_group=self.channels_per_group,
            n_blocks=4, nz_feat=nz_feat, bott_size=self.bottleneck_size)

        self.code_predictor = CodePredictor(
            norm_f0, nz_feat=nz_feat, nenc_feat=self.encoder.nenc_feat, betas_scale=betas_scale,
            use_smal_betas=True, shape_init=self.shape_init)
        self.return_feat = return_feat

    def forward(self, img, masks=None, is_optimization=False, is_var_opt=False):
        img_feat, enc_feat, feat_resnet, feat_multiscale = self.encoder.forward(img, masks)
        codes_pred = self.code_predictor.forward(img_feat, enc_feat)
        if self.return_feat:
            return codes_pred, feat_resnet, feat_multiscale
        else:
            return codes_pred


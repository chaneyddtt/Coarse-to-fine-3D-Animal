from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from util import config
import pickle as pkl
criterionL2 = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()


# L1 or L2 based sihouette loss
def mask_loss(mask_pred, mask_gt):
    # return torch.nn.L1Loss()(mask_pred, mask_gt)
    return criterionL2(mask_pred, mask_gt)


# 2D keypoint loss
def kp_l2_loss(kp_pred, kp_gt, num_joints):
    vis = (kp_gt[:, :, 2, None] > 0).float()
    if not kp_pred.ndim == 3:
        kp_pred.reshape((kp_pred.shape[0], num_joints, -1))
    return criterionL2(vis * kp_pred, vis * kp_gt[:, :, :2])


# compute shape prior based on Mahalanobis distance, formulation taken from
# https://github.com/benjiebob/SMALify/blob/master/smal_fitter/smal_fitter.py
class Shape_prior(torch.nn.Module):
    def __init__(self, prior, shape_family_id, device, data_path=None):
        '''
        Args:
            prior: specify the prior to use, smal or unity or self-defined data
            shape_family_id: specify animal category id
            device: specify device
            data_path: specify self-defined data path if do not use smal or unity
        '''
        super(Shape_prior, self).__init__()
        if prior == 'smal':
            nbetas=20
            with open(config.SMAL_DATA_FILE, 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            shape_cluster_means = data['cluster_means'][shape_family_id]
            betas_cov = data['cluster_cov'][shape_family_id]
            betas_mean = torch.from_numpy(shape_cluster_means).float().to(device)
        elif prior == 'unity':
            nbetas=26
            unity_data = np.load(config.UNITY_SHAPE_PRIOR)
            betas_cov = unity_data['cov'][:-1, :-1]
            betas_mean = torch.from_numpy(unity_data['mean'][:-1]).float().to(device)
        else:
            assert data_path is not None
            nbetas=26
            prior_data = np.load(data_path, allow_pickle=True)
            betas_mean = torch.from_numpy(prior_data.item()['mean']).float().to(device)
            betas_cov = prior_data.item()['cov']

        invcov = np.linalg.inv(betas_cov + 1e-5 * np.eye(betas_cov.shape[0]))
        prec = np.linalg.cholesky(invcov)
        self.betas_prec = torch.Tensor(prec)[:nbetas, :nbetas].to(device)
        self.betas_mean = betas_mean[:nbetas]

    def __call__(self, betas_pred):
        diff = betas_pred - self.betas_mean.unsqueeze(0)
        res = torch.tensordot(diff, self.betas_prec, dims=([1], [0]))
        return (res**2).mean()


# Laplacian loss, calculate the Laplacian coordiante of both coarse and refined vertices and then compare the difference
class Laplacian(torch.nn.Module):
    def __init__(self, adjmat, device):
        '''
        Args:
            adjmat: adjacency matrix of the input graph data
            device: specify device for training
        '''
        super(Laplacian, self).__init__()
        adjmat.data = np.ones_like(adjmat.data)
        adjmat = torch.from_numpy(adjmat.todense()).float()
        dg = torch.sum(adjmat, dim=-1)
        dg_m = torch.diag(dg)
        ls = dg_m - adjmat
        self.ls = ls.unsqueeze(0).to(device)  # Should be normalized by the diagonal elements according to
                                              # the origial definition, this one also works fine.

    def forward(self, verts_pred, verts_gt, smooth=False):
        verts_pred = torch.matmul(self.ls, verts_pred)
        verts_gt = torch.matmul(self.ls, verts_gt)
        loss = torch.norm(verts_pred - verts_gt, dim=-1).mean()
        if smooth:
            loss_smooth = torch.norm(torch.matmul(self.ls, verts_pred), dim=-1).mean()
            return loss, loss_smooth
        return loss, None

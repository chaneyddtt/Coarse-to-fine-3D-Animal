import torch
import torch.nn as nn
import numpy as np
from util.nmr import NeuralRenderer
from smal.smal_torch import SMAL
from model.smal_mesh_net_img import MeshNet_img
from util import config

unity_shape_prior = np.load('data/priors/unity_betas.npz')


class MeshModel(nn.Module):
    def __init__(self, device, shape_family_id, betas_scale=False, shape_init=None, render_rgb=False):
        '''
        Args:
            device: specify device for training
            shape_family_id: specify animal category id
            betas_scale: whether predict the additional shape parameter proposed by WLDO
            shape_init: whether intialize the bias with a mean shape, choose from smal or unity
            render_rgb: whether render 3D mesh into 2D to get rgb image. Only set to true when generating
            visualization to save inference time.
        '''

        super(MeshModel, self).__init__()

        self.model_renderer = NeuralRenderer(config.IMG_RES, proj_type=config.PROJECTION,
                                             norm_f0=config.NORM_F0,
                                             norm_f=config.NORM_F,
                                             norm_z=config.NORM_Z,
                                             render_rgb=render_rgb,
                                             device=device)
        self.model_renderer.directional_light_only()
        self.smal = SMAL(device, shape_family_id=shape_family_id)
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
                                   shape_init=shape_init)
        print('INITIALIZED')

    def forward(self, inp):

        pred_codes = self.meshnet(inp)

        return pred_codes
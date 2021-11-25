import pickle as pkl
import numpy as np
from chumpy import Ch
import cv2
import torch

name2id35 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0, 'LEar':33, 'REar':34}
id2name35 = {v: k for k, v in name2id35.items()}


class Prior(object):
    def __init__(self, prior_path, device):
        with open(prior_path, 'rb')  as f:
            res = pkl.load(f, encoding='latin1')

            self.mean_ch = res['mean_pose']
            self.precs_ch = res['pic']

            self.precs = torch.from_numpy(res['pic'].r.copy()).float().to(device)
            self.mean = torch.from_numpy(res['mean_pose']).float().to(device)

        prefix = 3
        pose_len = 105
        id2name = id2name35
        name2id = name2id35

        self.use_ind = np.ones(pose_len, dtype=bool)
        self.use_ind[:prefix] = False
        self.use_ind_tch = torch.from_numpy(self.use_ind).float().to(device)

    def __call__(self, x):
        mean_sub = x.reshape(-1, 35*3) - self.mean.unsqueeze(0)
        res = torch.tensordot(mean_sub, self.precs, dims=([1], [0])) * self.use_ind_tch
        return (res**2).mean()
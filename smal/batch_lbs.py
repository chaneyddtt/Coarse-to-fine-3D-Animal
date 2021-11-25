from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


def batch_skew(vec, batch_size=None, opts=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    if batch_size is None:
        batch_size = vec.shape.as_list()[0]
    col_inds = torch.LongTensor([1, 2, 3, 5, 6, 7])
    indices = torch.reshape(torch.reshape(torch.arange(0, batch_size) * 9, [-1, 1]) + col_inds, [-1, 1])
    updates = torch.reshape(
            torch.stack(
                [
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ],
                dim=1), [-1])
    out_shape = [batch_size * 9]
    res = torch.Tensor(np.zeros(out_shape[0])).cuda(device=vec.device)
    res[np.array(indices.flatten())] = updates
    res = torch.reshape(res, [batch_size, 3, 3])

    return res

def batch_rodrigues(theta, opts=None):
    """
    Theta is Nx3
    """
    batch_size = theta.shape[0]

    angle = (torch.norm(theta + 1e-8, p=2, dim=1)).unsqueeze(-1)
    r = (torch.div(theta, angle)).unsqueeze(-1)

    angle = angle.unsqueeze(-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    outer = torch.matmul(r, r.transpose(1,2))

    eyes = torch.eye(3).unsqueeze(0).repeat([batch_size, 1, 1]).cuda(device=theta.device)
    H = batch_skew(r, batch_size=batch_size, opts=opts)
    R = cos * eyes + (1 - cos) * outer + sin * H 

    return R

def batch_lrotmin(theta):
    """
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.


    Args:
      pose: `Tensor`, N x 72 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=24

    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 23=(K-1) joints with identity subtracted.,
    """
    # Ignore global rotation
    theta = theta[:,3:]

    Rs = batch_rodrigues(torch.reshape(theta, [-1,3]))
    lrotmin = torch.reshape(Rs - torch.eye(3), [-1, 207])

    return lrotmin

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False, betas_logscale=None, opts=None):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """
    if rotate_base:
        print('Flipping the SMPL coordinate frame!!!!')
        rot_x = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rot_x = torch.reshape(torch.repeat(rot_x, [N, 1]), [N, 3, 3]) # In tf it was tile
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)
    N = Rs.shape[0]

    Js_orig = Js.clone()

    scaling_factors = torch.ones(N, parent.shape[0], 3).to(Rs.device)
    if betas_logscale is not None:
        leg_joints = list(range(7,11)) + list(range(11,15)) + list(range(17,21)) + list(range(21,25))
        tail_joints = list(range(25, 32))
        ear_joints = [33, 34]

        beta_scale_mask = torch.zeros(35, 3, 6).to(betas_logscale.device)
        beta_scale_mask[leg_joints, [2], [0]] = 1.0 # Leg lengthening
        beta_scale_mask[leg_joints, [0], [1]] = 1.0 # Leg fatness
        beta_scale_mask[leg_joints, [1], [1]] = 1.0 # Leg fatness
        
        beta_scale_mask[tail_joints, [0], [2]] = 1.0 # Tail lengthening
        beta_scale_mask[tail_joints, [1], [3]] = 1.0 # Tail fatness
        beta_scale_mask[tail_joints, [2], [3]] = 1.0 # Tail fatness
        
        beta_scale_mask[ear_joints, [1], [4]] = 1.0 # Ear y
        beta_scale_mask[ear_joints, [2], [5]] = 1.0 # Ear z

        beta_scale_mask = torch.transpose(
            beta_scale_mask.reshape(35*3, 6), 0, 1)

        betas_scale = torch.exp(betas_logscale @ beta_scale_mask)
        scaling_factors = betas_scale.reshape(-1, 35, 3)

    scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)

    def make_A(R, t):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = torch.nn.functional.pad(R, (0,0,0,1,0,0))
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).to(Rs.device)], 1)
        return torch.cat([R_homo, t_homo], 2)
    
    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]

        s_par_inv = torch.inverse(scale_factors_3x3[:, parent[i]])
        rot = Rs[:, i]
        s = scale_factors_3x3[:, i]
        
        rot_new = s_par_inv @ rot @ s

        A_here = make_A(rot_new, j_here)
        res_here = torch.matmul(
            results[parent[i]], A_here)
        
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    # scale updates
    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = torch.cat([Js_orig, torch.zeros([N, 35, 1, 1]).to(Rs.device)], 2)
    init_bone = torch.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = torch.nn.functional.pad(init_bone, (3,0,0,0,0,0,0,0))
    A = results - init_bone

    return new_J, A

# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : <anning@cpl.ac.cn>
# @Author : Youjia Zhang   @Email : <zhangyoujia@cpl.ac.cn>
# @Author : Cong Lin       @Email : <lincong8722@gmail.com>
# @Author : Zhenyu Sun     @Email : <sunzhenyu@cpl.ac.cn>

import numpy as np
import torch
from utils.interp import resample_sphere_surface_barycentric


def apply_rotate_matrix(euler_angle, xyz_moving, norm=False, en=None, face=None):
    a = euler_angle[:, [0]]
    b = euler_angle[:, [1]]
    g = euler_angle[:, [2]]
    r1 = torch.cat(
        [torch.cos(g) * torch.cos(b), torch.cos(g) * torch.sin(b) * torch.sin(a) - torch.sin(g) * torch.cos(a),
         torch.sin(g) * torch.sin(a) + torch.cos(g) * torch.cos(a) * torch.sin(b)], dim=1)
    r2 = torch.cat(
        [torch.cos(b) * torch.sin(g), torch.cos(g) * torch.cos(a) + torch.sin(g) * torch.sin(b) * torch.sin(a),
         torch.cos(a) * torch.sin(g) * torch.sin(b) - torch.cos(g) * torch.sin(a)], dim=1)
    r3 = torch.cat(
        [-torch.sin(b), torch.cos(b) * torch.sin(a), torch.cos(b) * torch.cos(a)], dim=1)
    moved_x = torch.sum(xyz_moving * r1, dim=1, keepdim=True)
    moved_y = torch.sum(xyz_moving * r2, dim=1, keepdim=True)
    moved_z = torch.sum(xyz_moving * r3, dim=1, keepdim=True)
    sphere_moved = torch.cat((moved_x, moved_y, moved_z), dim=1)

    if norm:
        sphere_moved = sphere_moved / torch.norm(sphere_moved, dim=1, keepdim=True)
    return sphere_moved

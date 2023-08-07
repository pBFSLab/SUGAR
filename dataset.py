# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : <anning@cpl.ac.cn>
# @Author : Youjia Zhang   @Email : <zhangyoujia@cpl.ac.cn>
# @Author : Cong Lin       @Email : <lincong8722@gmail.com>
# @Author : Zhenyu Sun     @Email : <sunzhenyu@cpl.ac.cn>

import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from nibabel.freesurfer import read_morph_data, read_geometry, read_annot
from utils.rotate_matrix import apply_rotate_matrix
from utils.interp import resample_sphere_surface_barycentric


def normalize(data, norm_method=None):
    assert norm_method == 'zscore'
    data = (data - data.mean()) / data.std()

    index = np.where(data < -3)[0]
    data[index] = -3 - (1 - np.exp(3 - np.abs(data[index])))
    index = np.where(data > 3)[0]
    data[index] = 3 + (1 - np.exp(3 - np.abs(data[index])))

    return data


def data_random_rotate(sulc, _, xyz):
    euler = np.random.random(3) * 0.01
    euler_t = torch.from_numpy(euler.reshape(1, -1)).float().to('cuda')
    sulc_t = torch.from_numpy(sulc).float().to('cuda')
    xyz_t = torch.from_numpy(xyz).float().to('cuda')
    xyz_r = apply_rotate_matrix(euler_t, xyz_t, norm=True)
    sulc_r = resample_sphere_surface_barycentric(xyz_r, xyz_t, sulc_t.unsqueeze(1))
    return sulc_r.squeeze().cpu().numpy(), 0


class SphericalDataset(Dataset):
    def __init__(self, sublist=None, dir_fixed=None, dir_result=None,
                 lrh='lh', feature='sulc', norm_type='zscore',
                 ico_levels=None,
                 seg=False, is_train=True, is_da=False, is_rigid=False):
        self.dir_fixed = dir_fixed
        self.dir_result = dir_result
        self.lrh = lrh
        self.feature = feature
        self.norm_type = norm_type
        if isinstance(sublist, str):
            with open(sublist, "r") as f:
                self.sub_dirs = f.readlines()
        else:
            self.sub_dirs = sublist

        self.sulc_fixed = None

        self.ico_levels = ico_levels

        self.fixed = None

        self.seg = seg
        self.is_train = is_train
        self.is_rigid = is_rigid
        self.is_da = is_da

    def get_fixed(self):
        normalize_type = self.norm_type

        if self.fixed is None:
            fixed = list()
            for ico_level in self.ico_levels:
                dir_fixed = self.dir_fixed
                sulc_fixed = read_morph_data(os.path.join(dir_fixed, ico_level, 'surf', f'{self.lrh}.sulc')).astype(np.float32)
                xyz_fixed, faces_fixed = read_geometry(os.path.join(dir_fixed, ico_level, 'surf', f'{self.lrh}.sphere'))
                xyz_fixed = xyz_fixed.astype(np.float32) / 100
                faces_fixed = faces_fixed.astype(int)

                sulc_fixed = normalize(sulc_fixed, normalize_type)

                seg_fixed, seg_color_fixed, seg_name_fixed = read_annot(
                    os.path.join(dir_fixed, ico_level, 'label', f'{self.lrh}.aparc.annot'))
                seg_fixed = seg_fixed.astype(int)
                fixed.append([sulc_fixed, 0, xyz_fixed, faces_fixed, seg_fixed])
            self.fixed = fixed
        return self.fixed

    def get_moving(self, sub_id, ico_level):
        sub_dir_result = os.path.join(self.dir_result, sub_id, 'surf')
        if self.is_rigid:
            data_type = 'orig'
        else:
            data_type = 'rigid'

        sulc_moving_interp = os.path.join(sub_dir_result, f'{self.lrh}.{data_type}.interp_{ico_level}.sulc')
        sphere_moving_file = os.path.join(sub_dir_result, f'{self.lrh}.{data_type}.interp_{ico_level}.sphere')

        sulc_moving = read_morph_data(sulc_moving_interp).astype(np.float32)

        xyz_moving, faces_moving = read_geometry(sphere_moving_file)
        xyz_moving = xyz_moving.astype(np.float32) / 100
        faces_moving = faces_moving.astype(int)

        if self.is_da:
            sulc_moving, _ = data_random_rotate(sulc_moving, 0, xyz_moving)

        normalize_type = self.norm_type
        sulc_moving = normalize(sulc_moving, normalize_type)

        seg_moving = np.zeros_like(sulc_moving, dtype=int)

        return sulc_moving, 0, xyz_moving, faces_moving, seg_moving

    def __getitem__(self, index):
        sub_id = self.sub_dirs[index].strip()

        fixed = self.get_fixed()
        movings = list()

        for ico_level in self.ico_levels:
            sulc_moving, _, xyz_moving, faces_moving, seg_moving = self.get_moving(sub_id, ico_level)
            movings.append([sulc_moving, _, xyz_moving, faces_moving, seg_moving])

        return movings, fixed, sub_id

    def __len__(self):
        return len(self.sub_dirs)

# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : <anning@cpl.ac.cn>
# @Author : Youjia Zhang   @Email : <zhangyoujia@cpl.ac.cn>
# @Author : Cong Lin       @Email : <lincong8722@gmail.com>
# @Author : Zhenyu Sun     @Email : <sunzhenyu@cpl.ac.cn>

import os
import time

import nibabel as nib
import torch
from torch.utils.data import DataLoader

from dataset import SphericalDataset
from utils.interp import interp_sulc_barycentric
from utils.rotate_matrix import apply_rotate_matrix
from utils.interp import resample_sphere_surface_barycentric, upsample_std_sphere_torch
from utils.auxi_data import get_points_num_by_ico_level


def interp_dir_single(dir_recon: str, dir_rigid: str, dir_fixed: str, ico_level: str, hemis=None, is_rigid=False):
    if hemis is None:
        hemis = ['lh', 'rh']
    elif isinstance(hemis, str):
        hemis = [hemis]
    elif isinstance(hemis, list):
        pass

    surf_dir_recon = os.path.join(dir_recon, 'surf')
    surf_dir_rigid = os.path.join(dir_rigid, 'surf')
    if not os.path.exists(surf_dir_rigid):
        os.makedirs(surf_dir_rigid, exist_ok=True)
    for hemisphere in hemis:
        sphere_fixed_file = os.path.join(dir_fixed, ico_level, 'surf', f'{hemisphere}.sphere')

        sulc_moving_file = os.path.join(surf_dir_recon, f'{hemisphere}.sulc')
        if is_rigid:
            data_type = 'orig'
            sphere_moving_file = os.path.join(surf_dir_recon, f'{hemisphere}.sphere')
        else:
            data_type = 'rigid'
            sphere_moving_file = os.path.join(surf_dir_rigid, f'{hemisphere}.rigid.sphere')  # 跑完刚性配准以后有这个文件

        if not os.path.exists(sphere_moving_file):
            continue

        sulc_moving_interp_file = os.path.join(surf_dir_rigid, f'{hemisphere}.{data_type}.interp_{ico_level}.sulc')
        sphere_moving_interp_file = os.path.join(surf_dir_rigid,
                                                 f'{hemisphere}.{data_type}.interp_{ico_level}.sphere')
        interp_sulc_barycentric(sulc_moving_file, sphere_moving_file, sphere_fixed_file,
                                sulc_moving_interp_file,
                                sphere_moving_interp_file)
        print(f'interp: >>> {sulc_moving_interp_file}')
        print(f'interp: >>> {sphere_moving_interp_file}')


def save_sphere_reg(config, hemisphere, xyz_moved, euler_angle, dir_recon, dir_rigid, dir_result, device):
    # #################################################################################################### #
    faces = config['face']

    if config['is_rigid']:
        data_type = 'orig'
        surf_dir_out = os.path.join(dir_rigid, 'surf')
    else:
        data_type = 'rigid'
        surf_dir_out = os.path.join(dir_rigid, 'surf')

    # # save sphere.reg in f saverage6 not apply rotate matrix
    faces_fs = faces[config["ico_level"]].cpu().numpy()
    if not os.path.exists(surf_dir_out):
        os.makedirs(surf_dir_out)
    sphere_moved_fs_file = os.path.join(surf_dir_out,
                                        f'{hemisphere}.{data_type}.interp_{config["ico_level"]}.sphere.reg')
    nib.freesurfer.write_geometry(sphere_moved_fs_file, xyz_moved.detach().cpu().numpy() * 100, faces_fs)
    print(f'sphere_moved_fs_file >>> {sphere_moved_fs_file}')

    # interp sphere.reg to native space
    if config['is_rigid']:
        sphere_rigid_native_file = os.path.join(dir_recon, 'surf', f'{hemisphere}.sphere')
        sphere_moved_native_file = os.path.join(surf_dir_out, f'{hemisphere}.rigid.sphere')
        xyz_native, faces_native = nib.freesurfer.read_geometry(sphere_rigid_native_file)
        xyz_native = torch.from_numpy(xyz_native).float().to(device) / 100
        xyz_moved_native = apply_rotate_matrix(euler_angle, xyz_native, norm=True)

        nib.freesurfer.write_geometry(sphere_moved_native_file, xyz_moved_native.detach().cpu().numpy() * 100,
                                      faces_native)
        print(f'sphere.reg >>> {sphere_moved_native_file}')


def infer(moving_datas, fixed_datas, models, faces, ico_levels, features, device='cuda'):
    assert len(moving_datas) > 0

    sulc_moving_fs6, _, xyz_moving_fs6, faces_moving_fs6, seg_moving_fs6 = moving_datas
    sulc_fixed_fs6, _, xyz_fixed_fs6, faces_fixed_fs6, seg_fixed_fs6 = fixed_datas

    sulc_moving_fs6 = sulc_moving_fs6.T.to(device)
    sulc_fixed_fs6 = sulc_fixed_fs6.T.to(device)

    seg_moving_fs6 = seg_moving_fs6.squeeze().to(device)
    seg_fixed_fs6 = seg_fixed_fs6.squeeze().to(device)

    xyz_moving_fs6 = xyz_moving_fs6.squeeze().to(device)
    xyz_fixed_fs6 = xyz_fixed_fs6.squeeze().to(device)

    # NAMIC dont have 35
    if not torch.any(seg_moving_fs6 == 0):
        seg_fixed_fs6[seg_fixed_fs6 == 0] = 35

    # 904.et dont have 4
    if not torch.any(seg_moving_fs6 == 4):
        seg_fixed_fs6[seg_fixed_fs6 == 4] = 0

    xyz_moved = None
    seg_moving_lap = None
    for idx, model in enumerate(models):
        feature = features[idx]
        assert feature == 'sulc'
        data_moving_fs6 = sulc_moving_fs6.to(device)
        data_fixed_fs6 = sulc_fixed_fs6.to(device)

        ico_level = ico_levels[idx]
        points_num = get_points_num_by_ico_level(ico_level)
        faces_sphere = faces[ico_level].to(device)
        data_moving = data_moving_fs6[:points_num]
        data_fixed = data_fixed_fs6[:points_num]
        seg_moving = seg_moving_fs6[:points_num]
        seg_fixed = seg_fixed_fs6[:points_num]
        xyz_moving = xyz_moving_fs6[:points_num]
        xyz_fixed = xyz_fixed_fs6[:points_num]

        if xyz_moved is None:
            data_x = torch.cat((data_moving, data_fixed), 1).to(device)
            data_x = data_x.detach()

            xyz_moved_lap, euler_angle = model(data_x, xyz_moving, face=faces_sphere)

            xyz_moved = apply_rotate_matrix(euler_angle, xyz_moving, norm=True,
                                            en=model.en, face=faces_sphere)

            data_moving_lap = data_moving
        else:
            # upsample xyz_moved
            xyz_moved_upsample = upsample_std_sphere_torch(xyz_moved, norm=True)
            xyz_moved_upsample = xyz_moved_upsample.detach()

            # moved数据重采样
            moving_data_resample = resample_sphere_surface_barycentric(xyz_moved_upsample, xyz_fixed, data_moving)

            data_x = torch.cat((moving_data_resample, data_fixed), 1).to(device)

            xyz_moved_lap, euler_angle = model(data_x, xyz_moving, face=faces_sphere)

            euler_angle_interp_moved_upsample = resample_sphere_surface_barycentric(xyz_fixed, xyz_moved_upsample,
                                                                                    euler_angle)
            xyz_moved = apply_rotate_matrix(euler_angle_interp_moved_upsample, xyz_moved_upsample, norm=True,
                                            face=faces_sphere)

            data_moving_lap = moving_data_resample

    seg_fixed = False

    return xyz_fixed, xyz_moved, xyz_moved_lap, data_fixed, data_moving, data_moving_lap, euler_angle, \
        seg_moving, seg_moving_lap, seg_fixed


def run_epoch(models, faces, config, dataloader,
              save_result=False, dir_recon=None, dir_rigid=None, dir_result=None):
    device = config['device']
    features = config['feature']
    ico_levels = config['ico_levels']
    subs_loss = []
    for datas_moving, datas_fixed, _ in dataloader:
        datas_moving = datas_moving[0]

        datas_fixed = datas_fixed[0]

        time_s = time.time()
        xyz_fixed, xyz_moved, xyz_moved_lap, fixed_data, data_moving, data_moving_lap, euler_angle, \
            seg_moving, seg_moving_lap, seg_fixed, \
            = infer(datas_moving, datas_fixed, models, faces, ico_levels, features, device)
        print(f'all infer time: {time.time() - time_s: 0.4f}s')

        if save_result:
            hemisphere = config["hemisphere"]
            dir_recon = os.path.join(dir_recon, config['subjs'][0])
            dir_rigid = os.path.join(dir_rigid, config['subjs'][0])
            dir_result = os.path.join(dir_result, config['subjs'][0])
            save_sphere_reg(config, hemisphere, xyz_moved, euler_angle, dir_recon, dir_rigid, dir_result, device)

    return subs_loss


@torch.no_grad()
def hemisphere_predict(models, config, hemisphere,
                       dir_recon, dir_rigid=None, dir_result=None,
                       seg=False):
    for model in models:
        model.eval()
    # 获取config_train的配置
    feature = config['feature']  # 加载的数据类型
    faces = config['face']

    # 数据目录
    dir_fixed = config["dir_fixed"]  # fixed数据目录

    dataset_train = SphericalDataset(config['subjs'], dir_fixed, os.path.dirname(dir_rigid),
                                     hemisphere, feature=feature, norm_type=config["normalize_type"],
                                     ico_levels=['fsaverage6'],
                                     seg=seg, is_train=False, is_da=False, is_rigid=config['is_rigid'])

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, num_workers=0)

    subs_loss = run_epoch(models, faces, config, dataloader_train,
                          save_result=True, dir_recon=os.path.dirname(dir_recon),
                          dir_rigid=os.path.dirname(dir_rigid), dir_result=os.path.dirname(dir_result))

    return subs_loss


def train_val(config):
    # 获取config_train的配置
    device = config['device']  # 使用的硬件

    if config['validation'] is True:
        # 1. interp file
        interp_dir_single(config["dir_predict_recon"], config["dir_predict_rigid"], config["dir_fixed"],
                          'fsaverage6', hemis=config['hemisphere'], is_rigid=config['is_rigid'])

        models = []
        for model_file in config['model_files'][:config["ico_index"] + 1]:
            print(f'<<< model : {model_file}')
            model = torch.load(model_file)['model']
            model.to(device)
            model.eval()
            models.append(model)

        hemisphere_predict(models, config, config['hemisphere'],
                           dir_recon=config['dir_predict_recon'],
                           dir_rigid=config['dir_predict_rigid'],
                           dir_result=config['dir_predict_result'],
                           seg=False)

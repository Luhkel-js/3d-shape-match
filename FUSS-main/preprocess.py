import os, csv
import scipy.io as sio
import numpy as np
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch

from utils.geometry_util import laplacian_decomposition, get_operators
from utils.shape_util import read_shape, compute_geodesic_distmat, write_off


if __name__ == '__main__':

    # parse arguments 解析参数
    parser = ArgumentParser('Preprocess .off files')
    parser.add_argument('--data_root', required=True, help='data root contains /off sub-folder.')
    parser.add_argument('--n_eig', type=int, default=300, help='number of eigenvectors/values to compute.') #特征向量的个数
    parser.add_argument('--no_eig', action='store_true', help='no laplacian eigen-decomposition')  #如果指定该参数，则不进行拉普拉斯特征分解
    parser.add_argument('--no_normalize', action='store_true', help='no normalization of face area.') #如果指定该参数，则不进行面面积的归一化
    args = parser.parse_args()

    # sanity check 有效性检查
    data_root = args.data_root
    n_eig = args.n_eig
    no_eig = args.no_eig
    no_normalize = args.no_normalize
    assert n_eig > 0, f'Invalid n_eig: {n_eig}'  #确保 n_eig 大于 0，否则抛出 AssertionError 异常
    assert os.path.isdir(data_root), f'Invalid data root: {data_root}'

    if not no_eig:
        spectral_dir = os.path.join(data_root, 'diffusion')
        os.makedirs(spectral_dir, exist_ok=True)

    # read .off files
    off_files = sorted(glob(os.path.join(data_root, 'off', '*.off')))
    assert len(off_files) != 0 #确保找到了至少一个 .off 文件，否则抛出 AssertionError 异常。

    # read mesh info file
    header = ['file_name', 'mean_x', 'mean_y', 'mean_z', 'face_area']
    with open(os.path.join(data_root, 'mesh_info.csv'), 'w+', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for off_file in tqdm(off_files): #遍历所有的 .off 文件，并使用 tqdm 显示进度条
            verts, faces = read_shape(off_file)
            filename = os.path.basename(off_file)

            if not no_normalize: #如果 no_normalize 为 False，则进行面面积归一化
                # center shape
                verts_mean = np.mean(verts, axis=0)
                verts -= verts_mean #将顶点坐标减去均值，使形状居中。

                # normalize verts by sqrt face area 计算面的平方根面积
                old_sqrt_area = laplacian_decomposition(verts=verts, faces=faces, k=1)[-1]
                print(f'Old face sqrt area: {old_sqrt_area:.3f}')
                verts /= old_sqrt_area

                # save new verts and faces
                write_off(off_file, verts, faces)

                # write mesh info
                data = [filename[:-4], verts_mean[0], verts_mean[1], verts_mean[2],
                        old_sqrt_area]
                writer.writerow(data)

            if not no_eig:
                # recompute laplacian decomposition 调用 get_operators 函数计算拉普拉斯算子的特征分解，并将结果存储在 spectral_dir 目录中。
                get_operators(torch.from_numpy(verts).float(), torch.from_numpy(faces).long(),
                              k=n_eig, cache_dir=spectral_dir)

import numpy as np
from pathlib import  Path
import colmap_read_write_model as rw


def load_colmap_data(base_dir):
    model_path = str(base_dir / "sparse/0/")
    # 读取数据
    camdata, imdata, points3D = rw.read_model(model_path, ext=".bin")

    # 获取高 宽 focal，只有一个相机
    list_of_keys = list(camdata.keys())
    assert len(list_of_keys) == 1
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam))
    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h,w,f]).reshape([3,1])

    # 计算Tcw
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats

    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)

    # poses: [N,4,4], points3D: pointclouds, perm: [N], hwf: height-width-focal
    return poses, points3D, perm, hwf


def convert(basedir):
    # 读取colmap bin data
    # poses: c2w - Twc
    poses, pts3d, perm, hwf = load_colmap_data(basedir)

    # 所有sparse 3d points以及哪些image可见
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[0]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    Tcw = np.linalg.inv(poses)
    # 把点，从world转到camera系
    pts_arr_aug = np.concatenate([pts_arr, np.ones([pts_arr.shape[0], 1])], axis=1)
    # 获取camera系下的z坐标，也就是距离光心的深度
    zvals = (Tcw @ pts_arr_aug.T).transpose(2,0,1)[:,:,2]
    valid_z = zvals[vis_arr == 1]
    # 统计深度
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    # to 3x4 matrix
    poses = poses[:, :3, :]
    # concat hwf
    poses = np.concatenate([
        poses, np.tile(hwf[np.newaxis, ...], [poses.shape[0],1,1]),
    ], axis=2)

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)

        save_arr.append(np.concatenate([poses[i, ...].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)

    np.save( str(basedir / 'poses_bounds_my.npy'), save_arr)

    print( 'Done with imgs2poses' )




if __name__ == '__main__':
    basedir = Path("D:\\nerf\\colmapresult")
    convert(basedir)

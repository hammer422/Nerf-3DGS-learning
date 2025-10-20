import numpy as np
import math
import torch
import torch.nn as nn
from PIL import Image
import random

import colmap_read_write_model as colmaprw

from commondefine import *
from utils import *


def readDatasetFromColmap(source_path):
    image_folder = source_path / 'images'

    images = colmaprw.read_images_binary(
        str(source_path / "sparse" / "0" /  "images.bin")
    )
    cam_intrs = colmaprw.read_cameras_binary(
        str(source_path / "sparse" / "0" / "cameras.bin")
    )

    cam_infos_unsorted = []
    for key in images:
        extr = images[key]
        intr = cam_intrs[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # Tcw
        # 这里3dgs原始代码加了transpose
        # 这里我删掉了transpose，维持row-major
        R = colmaprw.qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_name = extr.name
        cam_info = CameraInfo(
            uid=uid, R=R, T=T,
            FovX=FovX, FovY=FovY,
            image_path=str(image_folder / image_name),
            image_name=str(image_name),
            width=width,
            height=height
        )
        cam_infos_unsorted.append(cam_info)

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    points3D = colmaprw.read_points3D_binary(
        str(source_path / "sparse" / "0" / "points3D.bin")
    )

    xyzs = []
    rgbs = []
    for _, point in points3D.items():
        xyzs.append(point.xyz)
        rgbs.append(point.rgb)

    xyzs = np.stack(xyzs, axis=0)
    rgbs = np.stack(rgbs, axis=0) / 255. # [0,1]
    normals = np.zeros_like(xyzs)

    pcd = BasicPointCloud(points=xyzs, colors=rgbs, normals=normals)

    nerf_normalization = getNerfppNorm(cam_infos)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=cam_infos,
        test_cameras=[],
        nerf_normalization=nerf_normalization
    )

    return scene_info



def create_camera_from_caminfo(cam_info: CameraInfo):
    image = Image.open(cam_info.image_path)

    cam = Camera(
        colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
        fovX=cam_info.FovX, fovY=cam_info.FovY,
        image=image, image_name=cam_info.image_name,
        width=image.size[0], height=image.size[1],
        is_test_dataset=False
    )
    cam.cuda()
    return cam



class Dataset:
    def __init__(self,
                config: Config,
                shuffle=True
                 ):
        self.config = Config
        self.scene_info = readDatasetFromColmap(
            config.source_path
        )

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)
            random.shuffle(self.scene_info.test_cameras)

        self.train_cameras = []
        self.test_cameras = []

        for cam_info in self.scene_info.train_cameras:
            self.train_cameras.append(
                create_camera_from_caminfo(cam_info)
            )

        for cam_info in self.scene_info.test_cameras:
            self.test_cameras.append(
                create_camera_from_caminfo(cam_info)
            )

    def get_pointcloud(self):
        return self.scene_info.point_cloud
    
    def get_nerf_normalization(self):
        return self.scene_info.nerf_normalization
    
    def get_train_cameras(self):
        return self.train_cameras
    
    def get_test_cameras(self):
        return self.test_cameras


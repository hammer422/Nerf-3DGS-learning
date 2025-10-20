from pathlib import Path
import numpy as np
from typing import NamedTuple
import torch.nn as nn

from utils import *



class Config:
    # common params
    source_path = Path("/home/zyb/workspace/3dgs-dataset")
    detect_anomaly = False
    is_eval = False
    eval_model_path = Path("")


    sh_degree = 3
    white_background = True
    train_test_exp = False # 是否启用曝光补偿

    iterations = 7000
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = iterations
    feature_lr = 0.0025
    opacity_lr = 0.025
    scaling_lr = 0.005
    rotation_lr = 0.001
    exposure_lr_init = 0.01
    exposure_lr_final = 0.0001
    exposure_lr_delay_steps = 5000 
    exposure_lr_delay_mult = 0.001
    percent_dense = 0.01
    lambda_dssim = 0.2



class CameraInfo(NamedTuple):
    uid: int
    R: np.array # Tcw
    T: np.array # Tcw
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict


class Camera(nn.Module):
    def __init__(self,
                colmap_id, R, T, fovX, fovY,
                image, image_name,
                width, height,
                is_test_dataset=False
                ):
        super().__init__()
        self.colmap_id = colmap_id # 和cam_info的uid相同
        self.R = R  # Tcw
        self.T = T  # Tcw
        self.fovX = fovX
        self.fovY = fovY
        self.image_name = image_name

        gt_image = PILtoTorch(image, (width, height))
        self.org_image = gt_image.clamp(0.0, 1.0)
        self.image_width = self.org_image.shape[2]
        self.image_height = self.org_image.shape[1]

        self.zfar = 100.0
        self.znear = 0.01


        self.is_test_dataset = is_test_dataset

        # 世界坐标系到相机坐标系
        self.world_view_transform = torch.Tensor(
            getWorld2View2(R, T)
        ).cuda()
        # view坐标系到NDC坐标系
        
        


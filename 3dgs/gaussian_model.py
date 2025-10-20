
import torch
import torch.nn as nn

from utils import *
from commondefine import *
from sh_utils import *


class GaussianModel:
    def __init__(self, sh_degree):
        self.sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0) # (N,3) 每个gaussian的空间位置
        self._features_dc = torch.empty(0) # (N,3) 球谐函数的0阶系数，也就是视角无关的基础颜色
        self._features_rest = torch.empty(0) # (N,(max_sh_degree+1)**2 - 1, 3) 球谐函数的高阶系数，建模视角相关的颜色变化
        self._scaling = torch.empty(0) # (N,3) 三个轴的缩放尺度
        self._rotation = torch.empty(0) # (N,4) 协方差的R，四元数
        self._opacity = torch.empty(0) # (N,1) 不透明度
        self.max_radii2D = torch.empty(0) # (N) 每个gaussian投影到2D图像的最大半径
        self.xyz_gradient_accum = torch.empty(0) # (N) 每个guassian的xyz梯度的累积
        self.denom = torch.empty(0) # (N) 每个gaussian被多少个视角观测到的count
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # helper function
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = lambda x: torch.log(x/(1-x)) # inverse_sigmoid
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def create_from_pcd(self,
            pcd: BasicPointCloud,
            cameras_list: list[Camera],
            spatial_lr_scale: float
        ):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0 # 这里实际没有拿到数据

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        
        # 最近邻的square distance用来初始化
        # 猜测：点密集-方差小， 点稀疏-方差大？？未经考证！
        dist2 = torch.clamp_min(distCUDA2_py(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 三个方向初始化，实质学习的参数是log的
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) 
        # R初始化无朝向，没有启发式的
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 不透明度为0.1， 然后计算回logit
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        """
        exposure为(C,3,4)，C为相机数量
        为什么需要曝光参数，视频序列不同图像可能因为自动曝光算法或者光线亮度白平衡，
        导致过曝或欠曝，3DGS会去试图用高斯的颜色来拟合这些光照，这是不合理的
        所以对每张图像引入曝光参数，建模成像系统的光照差异
        """
        self.exposure_mapping = {cam.image_name: idx for idx, cam in enumerate(cameras_list)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cameras_list), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

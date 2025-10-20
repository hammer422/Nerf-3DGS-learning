import socket
import sys
import random
import numpy as np
import torch
import datetime
import json

from pathlib import Path
from utils import *


def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


class Config:
    # common params
    source_path = Path("D:\\workspace\\3dgs-dataset")
    quiet = False
    detect_anomaly = False



    sh_degree = 3
    white_background = True

    iterations = 7000




class GaussianModel:
    def __init__(self, sh_degree, ):
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

    def create_from_pcd(self,
            pcd: BasicPointCloud,
            cam_infos,
            spatial_lr_scale
        ):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())


        print('d')



class Scene:
    def __init__(self, config: Config, gaussian_model: GaussianModel):
        self.gaussians = gaussian_model
        self.config = config
        self.scene_info = readDatasetFromColmap(
            config.source_path
        )
        self.gaussians.create_from_pcd(
            self.scene_info.point_cloud,
            self.scene_info.cam_infos,
            self.scene_info.nerf_normalization["radius"]
        )






class NetworkGUI:
    """
    - 接收客户端的请求
    - python端渲染出结果
    - send回客户端
    """
    def __init__(self, host="127.0.0.1", port=6009):
        self.host = host
        self.port = port
        self.conn = None
        self.addr = None
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)

    def try_connect(self):
        try:
            conn, addr = self.listener.accept()
            print(f"\nConnected by {addr}")
            conn.settimeout(None)
        except Exception as e:
            pass

    def read(self):
        messageLength = self.conn.recv(4)
        messageLength = int.from_bytes(messageLength, 'little')
        message = self.conn.recv(messageLength)
        return json.loads(message.decode("utf-8"))

    def receive(self):
        message = self.read()










def train(args: Config):

    network_gui = NetworkGUI()

    gaussian_model = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussian_model)


    start_iteration = 0

    for iteration in range(start_iteration, args.iterations+1):
        if network_gui.conn == None:
            # 尝试获取viewer
            network_gui.try_connect()






if __name__ == '__main__':
    args = Config()

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    train(args)




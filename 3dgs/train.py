import random


from dataset import Dataset
from gaussian_model import GaussianModel

from commondefine import *
from utils import *



def safe_state():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))






def train(args: Config):
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(args.sh_degree)
    scene = Dataset(args)
    
    # 根据dataset初始化gaussian model
    gaussians.create_from_pcd(
        scene.get_pointcloud(),
        scene.get_train_cameras(),
        scene.get_nerf_normalization["radius"]
    )

    # 训练setup
    gaussians.training_setup(args)

    # 启动iteration 如果resume则要修改这个iteration
    start_iteration = 0

    # 训练图像
    viewpoint_stack = scene.get_train_cameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    
    for iteration in range(start_iteration, args.iterations+1):
        # if network_gui.conn == None:
        #     # 尝试获取viewer
        #     network_gui.try_connect()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()


        # 随机选择一个camera训练
        if not viewpoint_stack:
            # empty
            viewpoint_stack = scene.get_train_cameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        viewpoint_rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(viewpoint_rand_idx)
        viewpoint_indices.pop(viewpoint_rand_idx)

        bg = background
        render_result = run_render(

        )








if __name__ == '__main__':
    args = Config()

    safe_state()

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    train(args)



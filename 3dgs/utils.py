import numpy as np
import math
import torch


def getWorld2View2(R, t):
    Tcw = np.zeros((4, 4))
    Tcw[:3, :3] = R
    Tcw[:3, 3] = t
    Tcw[3, 3] = 1.0

    return np.float32(Tcw)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T) # Tcw
        C2W = np.linalg.inv(W2C)  # Twc
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def PILtoTorch(pil_image, resolution):
    """
    PIL转torch
    转到[0,1]
    c,h,w
    """
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)



def distCUDA2_py(points: torch.Tensor, min_eps=0.0000001):
    """
    找到最近的3个点
    统计3个点的平方距离
    均值
    """
    N = points.shape[0]
    K = 3

    dist2 = torch.cdist(points, points, p=2) ** 2

    dist2[torch.arange(N), torch.arange(N)] = float("inf")

    knn_dists, _ = torch.topk(dist2, K, dim=1, largest=False)

    mean_dist2 = knn_dists.mean(dim=1)

    return mean_dist2


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def plot_expon_lr_func(
    lr_init, lr_final, lr_delay_steps, lr_delay_mult,
    max_steps, title="lr"
):
    from matplotlib import pyplot as plt

    lr_func = get_expon_lr_func(
        lr_init=lr_init,
        lr_final=lr_final,
        lr_delay_steps=lr_delay_steps,
        lr_delay_mult=lr_delay_mult,
        max_steps=max_steps
    )

    steps = np.arange(0, max_steps + 1)
    lrs = np.array([lr_func(step) for step in steps])

    plt.plot(steps, lrs, color='blue', linewidth=2)


    plt.show()
    plt.savefig(f"{title}.png")
    plt.close()


if __name__ == '__main__':

    # readDatasetFromColmap(Path("/home/zyb/workspace/3dgs-dataset"))

    # plot_expon_lr_func(
    #     0.00016, 0.0000016,
    #     0, 0.01, 7000, title="xyz_scheduler"
    # )

    # plot_expon_lr_func(
    #     0.01, 0.001,
    #     0, 0.0, 7000, title="exposure_scheduler"
    # )

    from torch.utils.cpp_extension import include_paths
    print(include_paths())
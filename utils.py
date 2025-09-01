import numpy as np

COLOR_CAMERA_MODEL = {
    "height": 720,
    "width": 1280,
    "fx": 691.34,
    "fy": 691.34,
    "cx": 636.641,
    "cy": 359.796,
    "k1": 0.00498968,
    "k2": -0.0475866,
    "k3": 0.0327495,
    "k4": 0,
    "k5": 0,
    "k6": 0,
    "p1": 0.00111541,
    "p2": 0.000362507
}

def construct_cam_intr():
    return {
        "K": np.array([
            [COLOR_CAMERA_MODEL["fx"], 0, COLOR_CAMERA_MODEL["cx"]],
            [0, COLOR_CAMERA_MODEL["fy"], COLOR_CAMERA_MODEL["cy"]],
            [0,0,1]
        ]),
        "dist": np.array([
            COLOR_CAMERA_MODEL["k1"], COLOR_CAMERA_MODEL["k2"],
            COLOR_CAMERA_MODEL["p1"], COLOR_CAMERA_MODEL["p2"],
            COLOR_CAMERA_MODEL["k3"]
        ]),
        "H": COLOR_CAMERA_MODEL["height"],
        "W": COLOR_CAMERA_MODEL["width"],
    }


if __name__ == '__main__':
    # for k in ["fx", "fy", "cx", "cy",
    #           "k1", "k2", "p1", "p2",
    #           "k3", "k4", "k5", "k6"]:
    #     print(f"{COLOR_CAMERA_MODEL[k]},", end="")

    pass


    # test ndc rays
    import torch
    def ndc_rays_opencv(H, W, focal, near, rays_o, rays_d):
        # 将原点推到近平面 (OpenCV: z forward +)
        t = (near - rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # 对称视锥: r=W/2, t=H/2
        o0 = +1.0 / (W / (2.0 * focal)) * (rays_o[..., 0] / rays_o[..., 2])
        o1 = -1.0 / (H / (2.0 * focal)) * (rays_o[..., 1] / rays_o[..., 2])
        o2 =  1.0 - 2.0 * near / rays_o[..., 2]

        d0 = +1.0 / (W / (2.0 * focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
        d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
        d2 = +2.0 * near / rays_o[..., 2]

        rays_o_ndc = torch.stack([o0, o1, o2], -1)
        rays_d_ndc = torch.stack([d0, d1, d2], -1)
        return rays_o_ndc, rays_d_ndc


    H, W = 800, 1200
    f = 800.0  # focal length
    near = f   # near plane = focal

    # 构造几条测试射线：中心点、左右边界、上下边界
    dirs = torch.tensor([
        [0.0, 0.0, 1.0],                # 中心
        [ (W/2)/f, 0.0, 1.0],           # 右边界
        [-(W/2)/f, 0.0, 1.0],           # 左边界
        [0.0, -(H/2)/f, 1.0],           # 上边界 (OpenCV: y向下)
        [0.0,  (H/2)/f, 1.0],           # 下边界
    ], dtype=torch.float32)

    origins = torch.zeros_like(dirs)

    o_ndc, d_ndc = ndc_rays_opencv(H, W, f, near, origins, dirs)
    out = torch.cat([o_ndc, d_ndc], dim=1).numpy()

    labels = ["center","right","left","top","bottom"]
    print("Columns: [o_x, o_y, o_z, d_x, d_y, d_z] (NDC)")
    for lab, row in zip(labels, out):
        print(f"{lab:7s}: {row}")


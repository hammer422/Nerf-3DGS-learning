# import numpy as np



# near = 0.2
# far = 50
# zsign = 1.0

# project_matrix = np.array([
#     [near, 0,0,0],
#     [0,near,0,0],
#     [0,0,near+far, -near*far],
#     [0,0,1,0],
# ])


# np.set_printoptions(suppress=True, precision=5)

# # 测试near plane上的点
# P1 = np.array([0,0,near,1])
# PS = project_matrix @ P1
# PS2 = PS / PS[-1]
# print(PS)
# print(PS2)


# # 测试far plane上的点
# P1 = np.array([0,2,far,1])
# PS = project_matrix @ P1
# PS2 = PS / PS[-1]
# print(PS)
# print(PS2)




import numpy as np
np.set_printoptions(precision=4, suppress=True)

near = 0.2
far = 100.

M_pesrp_to_ortho = np.array([
    [near, 0,0,0],
    [0,near,0,0],
    [0,0,-(far+near), -near*far],
    [0,0,-1,0]
])

P = np.array([1,1, -0.2, 1.])
P_NEW = M_pesrp_to_ortho @ P
P_NEW = P_NEW / P_NEW[-1]

print(P_NEW)





# np.random.seed(42)

# for _ in range(100):
#     l, r, n, f, t, b = np.random.random(6)

#     # M_ortho_trans = np.array([
#     #     [1,0,0,-(r+l)/2],
#     #     [0,1,0,-(t+b)/2],
#     #     [0,0,1,(n+f)/2],
#     #     [0,0,0,1]
#     # ])
#     # M_ortho_scale = np.array([
#     #     [2/(r-l),0,0,0],
#     #     [0,2/(t-b),0,0],
#     #     [0,0,2/(f-n),0],
#     #     [0,0,0,1]
#     # ])
#     # M_scale_trans_gt = np.array([
#     #     [2/(r-l),0,0,-(r+l)/(r-l)],
#     #     [0,2/(t-b),0,-(t+b)/(t-b)],
#     #     [0,0,2/(f-n),-(n+f)/(n-f)],
#     #     [0,0,0,1]
#     # ])
#     # M_scale_trans_hat = M_ortho_scale @ M_ortho_trans 

#     # if not np.allclose(M_scale_trans_gt, M_scale_trans_hat, atol=1e-3):
#     #     print(M_ortho_scale)
#     #     print(M_ortho_trans)
#     #     print(M_scale_trans_gt)
#     #     print(M_scale_trans_hat)
#     #     break

#     M_persp_to_ortho = np.array([
#         [n,0,0,0],
#         [0,n,0,0],
#         [0,0,-(f+n), -n*f],
#         [0,0,-1,0],
#     ])
#     M_ortho_scale_trans = np.array([
#         [2/(r-l), 0,0, -(r+l)/(r-l)],
#         [0,2/(t-b), 0, -(t+b)/(t-b)],
#         [0,0,2/(f-n), -(n+f)/(n-f)],
#         [0,0,0,1]
#     ])
#     M_full_gt = np.array([
#         [2*n/(r-l), 0, (r+l)/(r-l), 0],
#         [0, 2*n/(t-b), (t+b)/(t-b), 0],
#         [0,0, 3*(n+f)/(n-f), -2*n*f/(f-n)],
#         [0,0,-1,0]
#     ])
#     M_full = M_ortho_scale_trans @ M_persp_to_ortho


#     if not np.allclose(M_full, M_full_gt, atol=1e-4):
#         print(M_ortho_scale)
#         print(M_ortho_trans)
#         print(M_scale_trans_gt)
#         print(M_scale_trans_hat)
#         break



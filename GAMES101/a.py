import numpy as np

n = 1.
f = 5.
r = 2
l = -r
t = 2
b = -t


M_persp2ortho = np.array([
    [-n,0,0,0],
    [0,-n,0,0],
    [0,0,-(n+f), -n*f],
    [0,0,1,0]
])

M_ortho_corner = np.array([
    [1,0,0,-l],
    [0,1,0,-b],
    [0,0,1,n],
    [0,0,0,1]
])
M_ortho_scale = np.array([
    [2/(r-l),0,0,0],
    [0,2/(t-b),0,0],
    [0,0,2/(n-f),0],
    [0,0,0,1]
])
M_ortho_center = np.array([
    [1,0,0,-1],
    [0,1,0,-1],
    [0,0,1,-1],
    [0,0,0,1]
])
P1 = np.array([0,0,-n,1])
P2 = np.array([0,0,-f,1])

print(M_persp2ortho @ P1.T)
print(M_persp2ortho @ P2.T)

# print((-(n+f)*-f + n*f) / -f)

P1 = M_persp2ortho @ P1.T
# P1 /= P1[3]
P2 = M_persp2ortho @ P2.T
# P2 /= P2[3]

print(M_ortho_center @ M_ortho_scale @ M_ortho_corner @ P1)
print(M_ortho_center @ M_ortho_scale @ M_ortho_corner @ P2)




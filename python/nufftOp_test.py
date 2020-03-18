"""Script to test gpuNUFFT wrapper."""


import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time
from gpuNUFFT import NUFFTOp


# IMAGE
res = [64, 128]
[x, y] = np.meshgrid(np.linspace(-1, 1, res[0]),
                        np.linspace(-1, 1, res[1]))
img = (x**2 + y**2  < 0.5**2).T
img = img.astype(np.complex64)
plt.figure(1)
plt.imshow(abs(img[...]), aspect='equal')
plt.title('image')
plt.show()
print('Input image shape is', img.shape)


# KCOORDS
R = 1
n_lines = res[1] // R
ns_per_line = res[0]
kcoords = np.ones([ns_per_line * n_lines, 2])
readout_line = np.linspace(-0.5, 0.5, ns_per_line)
kcoords[:, 0] = np.matlib.repmat(readout_line, 1, n_lines)
kcoords[:, 1] = np.matlib.repmat(np.linspace(-0.5, 0.5, n_lines), ns_per_line, 1).T.reshape(-1)
print('Input kcoords shape is', kcoords.shape)


# WEIGHTS
weights = np.ones(kcoords.shape[0])
print('Input weights shape is', weights.shape)


# COIL MAPS
n_coils = 2
x, y = np.meshgrid(np.linspace(0, 1, res[0]), np.linspace(0, 1, res[1]))
coil_maps_1 = ((1 / (x**2 + y**2 + 1)).T).astype(np.complex64)
coil_maps_2 = np.flip(np.flip(coil_maps_1, axis=1), axis=0)
multi_img = np.tile(img, (n_coils, 1, 1))


if n_coils == 1:
    coil_maps = np.expand_dims(coil_maps_1, axis=0)
    plt.imshow(abs(coil_maps_1), aspect='equal')
    plt.title('coil map')
    plt.show()
elif n_coils == 2:
    coil_maps = np.stack([coil_maps_1, coil_maps_2])
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(abs(coil_maps_1), aspect='equal')
    axs[0].set_title('coil map 1')
    axs[1].imshow(abs(coil_maps_2), aspect='equal')
    axs[1].set_title('coil map 2')
    plt.show()
else:
    coil_maps = []
try:
    print('Input coil maps shape is', coil_maps.shape)
except:
    print('no coil maps')
# CALL NUFFT
wg = 3
sw = 8
osf = 2
balance_workload = True
print('-------------------------------')
print('Call NUFFTOp')
A = NUFFTOp(
    np.reshape(kcoords, kcoords.shape[::-1], order='F'),
    res,
    n_coils,
    None,
    weights,
    wg,
    sw,
    osf,
    balance_workload,
)
print('-------------------------------')
print('Apply forward op')
x = A.op(np.reshape(img.T, img.size))
x = A.op(np.asarray(
    [np.reshape(image_ch.T, image_ch.size) for image_ch in multi_img]
).T)
print('Output kdata shape is', x.shape)
print('-------------------------------')
print('Apply adjoint op')
img_adj = A.adj_op(x)
print('Output adjoint img shape is', img_adj.shape)
img_adj = np.squeeze(img_adj)
print(img_adj.shape)
plt.figure(3)
plt.imshow(abs(img_adj))#, cmap='gray')
plt.title('adjoint image')
plt.show()
print('done')
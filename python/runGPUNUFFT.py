from gpuNUFFT import NUFFTOp
import numpy as np
from pysap.data import get_sample_data
from mri.operators import NonCartesianFFT
from mri.operators.utils import convert_mask_to_locations, normalize_frequency_locations

# 2D MRI
radial_mask = get_sample_data("mri-radial-samples")
image = get_sample_data("2d-mri")
img_size = (512, 512)
weights = np.ones(radial_mask.shape[0])
f_op = NonCartesianFFT(samples=radial_mask.data, shape=img_size)
A = NUFFTOp(radial_mask.data, img_size, 1, weights, 3, 8, 2, 1)
x = A.op(image.data)
y = f_op.op(image.data)
img_back = A.adj_op(x)

# 2D pMRI
radial_mask = get_sample_data("mri-radial-samples")
image = get_sample_data("2d-pmri")
img_size = (512, 512)
weights = np.ones(radial_mask.shape[0])
f_op = NonCartesianFFT(samples=radial_mask.data, shape=img_size, n_coils=32)
A = NUFFTOp(radial_mask.data, img_size, image.shape[0], weights, 3, 8, 2, 1)
x = A.op(np.moveaxis(image, 0, -1))
x2 = A.op(np.asarray([np.reshape(channel_image.T, channel_image.size) for channel_image in image]).T)
y = f_op.op(image)
img_back = A.adj_op(x)

# 3D MRI
radial_mask = get_sample_data("mri-radial-3d-samples")
radial_mask = normalize_frequency_locations(radial_mask.data)
image = get_sample_data("3d-pmri")
image = image.data[0:2]
weights = np.ones(radial_mask.shape[0])
B = NUFFTOp(radial_mask, image.shape[1:], image.shape[0], weights, 3, 8, 2, 1)
f_op = NonCartesianFFT(samples=radial_mask, shape=image.shape[1:], n_coils=image.shape[0])
x = B.op(np.asarray([np.reshape(channel_image.T, channel_image.size) for channel_image in image]).T)
y = f_op.op(image)
img_back = B.adj_op(x)
img_back2 = f_op.adj_op(y)
img_back




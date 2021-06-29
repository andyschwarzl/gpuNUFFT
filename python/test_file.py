import numpy as np
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation


traj = np.load('/volatile/temp_traj.npy')
#D = estimate_density_compensation(traj, (384, 384, 208), 2)
fourier = NonCartesianFFT(traj, (384, 384, 208), 'gpuNUFFT')
K = fourier.op(np.zeros((384, 384, 208)))
print("Forward done")
I = fourier.adj_op(K)
print("Backward done")

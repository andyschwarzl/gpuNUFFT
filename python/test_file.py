import numpy as np
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation


traj = np.load('/volatile/temp_traj.npy')
for i in range(5):
    print(i)
    fourier = NonCartesianFFT(traj, (384, 384, 208), 'gpuNUFFT', n_coils=10, smaps=np.ones((10, 384, 384, 208)))
    fourier.impl.operator.set_smaps(np.ones((10, 384, 384, 208))+1)
    K = fourier.op(np.zeros((384, 384, 208)))
    im = fourier.adj_op(K)
    del fourier

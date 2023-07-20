import numpy as np
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
traj = np.load('/volatile/temp_traj.npy')


for i in range(1):
    dens = estimate_density_compensation(traj, (384, 384, 208))
    fourier = NonCartesianFFT(traj, (384, 384, 208), 'gpuNUFFT', n_coils=20, smaps=np.ones((20, 384, 384, 208)), osf=2, density_comp=dens)
    print(i)
    K = fourier.op(np.zeros((384, 384, 208)))
    I = fourier.adj_op(K)
    del fourier

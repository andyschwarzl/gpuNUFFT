import numpy as np
from mri.operators import NonCartesianFFT

traj = np.load('/volatile/temp_traj.npy')

fourier = NonCartesianFFT(traj, (384, 384, 208), 'gpuNUFFT', n_coils=20, smaps=np.ones((20, 384, 384, 208)), osf=1)

for i in range(10):
    print(i)
    K = fourier.op(np.zeros((384, 384, 208)))
    I = fourier.adj_op(K)
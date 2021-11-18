import numpy as np
from mri.operators import NonCartesianFFT

traj = np.load('/volatile/temp_traj.npy')

fourier = NonCartesianFFT(traj, (384, 384, 208), 'gpuNUFFT', n_coils=2, smaps=np.ones((2, 384, 384, 208)), osf=1)

for i in range(3):
    print(i)
    K = fourier.op(np.zeros((384, 384, 208)))
    I = fourier.adj_op(K)

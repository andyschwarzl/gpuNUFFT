import numpy as np
from mri.operators import NonCartesianFFT

traj = np.load('/volatile/temp_traj.npy')

for i in range(1):
    print(i)
    fourier = NonCartesianFFT(traj, (384, 384, 208), 'gpuNUFFT', n_coils=4, smaps=np.ones((4, 384, 384, 208)), osf=1)
    K = fourier.op(np.zeros((384, 384, 208)))

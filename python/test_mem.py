"""Script to test gpuNUFFT wrapper.
Authors:
Chaithya G R <chaithyagr@gmail.com>
"""

import numpy as np
from gpuNUFFT import NUFFTOp, MemoryAllocationType
import pytest


def test_memory_allocation_types():
    kspace_loc = np.random.random((5000, 3)) - 0.5
    img_size = [256, 256, 256]
    n_coils = 1
    image = np.random.random(img_size) + 1j * np.random.random(img_size)
    kspace = np.random.random((n_coils, kspace_loc.shape[0])) + 1j * np.random.random((n_coils, kspace_loc.shape[0]))
    kspace_out = []
    images_out = []
    nufft_ops = []
    for mem_allocation_type in list(MemoryAllocationType.__members__.values()):
        nufft_ops.append(NUFFTOp(
            kspace_loc=np.reshape(kspace_loc, kspace_loc.shape[::-1], order='F').astype(np.float32),
            image_size=img_size,
            num_coils=n_coils,
            when_allocate_memory=mem_allocation_type,
        ))
        kspace_out.append(nufft_ops[-1].op(input_image=image))
        images_out.append(nufft_ops[-1].adj_op(input_kspace=kspace))
        if len(nufft_ops) > 1:
            del nufft_ops[-2]
    kspace_out
    images_out
    images_out
    
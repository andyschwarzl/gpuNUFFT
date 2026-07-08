"""Script to test gpuNUFFT wrapper.
Authors:
Chaithya G R <chaithyagr@gmail.com>
"""

import numpy as np
from gpuNUFFT import NUFFTOp
import pytest




def test_memory_allocation_types():
    kspace_loc = np.random.random((5000, 3)) - 0.5
    img_size = [256, 256, 256]
    n_coils = 1
    image = np.random.random(img_size) + 1j * np.random.random(img_size)
    kspace = np.random.random((n_coils, kspace_loc.shape[0])) + 1j * np.random.random((n_coils, kspace_loc.shape[0]))
    kspace_out = []
    images_out = []
    nufft_op = []
    for mem_allocation_type in list(MemoryAllocationType.__members__.values()):
        nufft_op = NUFFTOp(
            kspace_loc=np.reshape(kspace_loc, kspace_loc.shape[::-1], order='F').astype(np.float32),
            image_size=img_size,
            num_coils=n_coils,
            when_allocate_memory=mem_allocation_type,
        )
        kspace_out.append(nufft_op.op(input_image=image))
        images_out.append(nufft_op.adj_op(input_kspace=kspace))
        del nufft_op
    kspace_out
    images_out
    images_out
    
    
def test_pinned_memory_provided():
    import cupyx as cpx
    
    kspace_loc = np.random.random((5000, 3)) - 0.5
    img_size = [256, 256, 256]
    n_coils = 1
    image = (np.random.random(img_size) + 1j * np.random.random(img_size)).astype(np.complex64)
    kspace = (np.random.random((n_coils, kspace_loc.shape[0])) + 1j * np.random.random((n_coils, kspace_loc.shape[0]))).astype(np.complex64)
    
    image_out = cpx.zeros_like_pinned(image)
    kspace_out = cpx.zeros_like_pinned(kspace)
    print("Addresses: ", hex(kspace_out.ctypes.data), hex(image_out.ctypes.data))
 
    nufft_op = NUFFTOp(
        kspace_loc=np.reshape(kspace_loc, kspace_loc.shape[::-1], order='F').astype(np.float32),
        image_size=img_size,
        num_coils=n_coils,
    )
    out_ksp = nufft_op.op(image, kspace_out)
    out_im = nufft_op.adj_op(in_kspace=kspace, out_image=image_out)
    out_ksp
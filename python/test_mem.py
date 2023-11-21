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
    image = np.random.random(img_size) + 1j * np.random.random(img_size)
    kspace = np.random.random((n_coils, kspace_loc.shape[0])) + 1j * np.random.random((n_coils, kspace_loc.shape[0]))
    
    image_out = cpx.empty_like_pinned(image)
    kspace_out = cpx.empty_like_pinned(kspace)
    
    nufft_ori = NUFFTOp(
        kspace_loc=np.reshape(kspace_loc, kspace_loc.shape[::-1], order='F').astype(np.float32),
        image_size=img_size,
        num_coils=n_coils,
        when_allocate_memory=MemoryAllocationType.ALLOCATE_MEMORY_IN_OP,
    )
    ori_kspace_out = nufft_ori.op(input_image=image)
    ori_image_out = nufft_ori.adj_op(input_kspace=kspace)
    
    nufft_op = NUFFTOp(
        kspace_loc=np.reshape(kspace_loc, kspace_loc.shape[::-1], order='F').astype(np.float32),
        image_size=img_size,
        num_coils=n_coils,
        when_allocate_memory=MemoryAllocationType.NEVER_ALLOCATE_MEMORY,
    )
    out_ksp = nufft_op.op(input_image=image, out_kspace=kspace_out)
    out_im = nufft_op.adj_op(input_kspace=kspace, out_image=image_out)
    out_ksp
"""Script to test gpuNUFFT wrapper.
Authors:
Chaithya G R <chaithyagr@gmail.com>
Carole Lazarus <carole.m.lazarus@gmail.com>
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from gpuNUFFT import NUFFTOp
import unittest


class TestgpuNUFFT(unittest.TestCase):
    """ Test the adjoint operator of the Wavelets both for 2D and 3D.
    """

    def get_nufft_op(self, sens_maps=None):
        return NUFFTOp(
            np.reshape(self.kspace_loc, self.kspace_loc.shape[::-1], order='F'),
            self.img_size,
            self.n_coils,
            sens_maps,
            self.weights,
            3,
            8,
            2,
            True,
        )

    def setUp(self):
        """ Setup variables:
        N = Image size
        max_iter = Number of iterations to test
        num_channels = Number of channels to be tested with for
                        multichannel tests
        """
        # IMAGE
        self.img_size = [64, 128]
        [x, y] = np.meshgrid(np.linspace(-1, 1, self.img_size[0]),
                                np.linspace(-1, 1, self.img_size[1]))
        img = (x**2 + y**2  < 0.5**2).T
        self.img = img.astype(np.complex64)
        plt.figure(1)
        plt.imshow(abs(img[...]), aspect='equal')
        plt.title('image')
        plt.show()
        print('Input image shape is', img.shape)

        # KCOORDS
        R = 1
        n_lines = self.img_size[1] // R
        ns_per_line = self.img_size[0]
        self.kspace_loc = np.ones([ns_per_line * n_lines, 2])
        readout_line = np.linspace(-0.5, 0.5, ns_per_line)
        self.kspace_loc[:, 0] = np.matlib.repmat(readout_line, 1, n_lines)
        self.kspace_loc[:, 1] = np.matlib.repmat(np.linspace(-0.5, 0.5, n_lines), ns_per_line, 1).T.reshape(-1)
        print('Input kcoords shape is', self.kspace_loc.shape)

        # WEIGHTS
        self.weights = np.ones(self.kspace_loc.shape[0])
        print('Input weights shape is', self.weights.shape)

        # COIL MAPS
        self.n_coils = 2
        x, y = np.meshgrid(np.linspace(0, 1, self.img_size[0]), np.linspace(0, 1, self.img_size[1]))
        coil_maps_1 = ((1 / (x**2 + y**2 + 1)).T).astype(np.complex64)
        coil_maps_2 = np.flip(np.flip(coil_maps_1, axis=1), axis=0)
        self.multi_img = np.tile(img, (self.n_coils, 1, 1))
        if self.n_coils == 1:
            self.coil_maps = np.expand_dims(coil_maps_1, axis=0)
            plt.imshow(abs(coil_maps_1), aspect='equal')
            plt.title('coil map')
            plt.show()
        elif self.n_coils == 2:
            self.coil_maps = np.stack([coil_maps_1, coil_maps_2])
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(abs(coil_maps_1), aspect='equal')
            axs[0].set_title('coil map 1')
            axs[1].imshow(abs(coil_maps_2), aspect='equal')
            axs[1].set_title('coil map 2')
            plt.show()

    def test_multicoil_with_sense(self):
        print('Apply forward op')
        operator = self.get_nufft_op(self.coil_maps)
        x = operator.op(np.reshape(self.img.T, self.img.size), False)
        y = np.random.random(x.shape)
        print('Output kdata shape is', x.shape)
        print('-------------------------------')
        print('Apply adjoint op')
        img_adj = operator.adj_op(x, False)
        adj_y = operator.adj_op(y, False)
        print('Output adjoint img shape is', img_adj.shape)
        img_adj = np.squeeze(img_adj).T
        adj_y = np.squeeze(adj_y).T
        print(img_adj.shape)
        plt.figure(3)
        plt.imshow(abs(img_adj))
        plt.title('adjoint image')
        plt.show()
        # Test Adjoint property
        x_d = np.vdot(self.img, adj_y)
        x_ad = np.vdot(x, y)
        np.testing.assert_allclose(x_d, x_ad, rtol=1e-5)
        print('done')

    def test_multicoil_without_sense(self):
        print('Apply forward op')
        operator = self.get_nufft_op()
        x = operator.op(np.asarray(
            [np.reshape(image_ch.T, image_ch.size) for image_ch in self.multi_img]
        ).T, False)
        y = np.random.random(x.shape)
        print('Output kdata shape is', x.shape)
        print('-------------------------------')
        print('Apply adjoint op')
        img_adj = operator.adj_op(x, False)
        print('Output adjoint img shape is', img_adj.shape)
        img_adj = np.squeeze(img_adj)
        img_adj = np.asarray(
                [image_ch.T for image_ch in img_adj]
            )
        adj_y = np.squeeze(operator.adj_op(y), False)
        adj_y = np.asarray(
                [image_ch.T for image_ch in adj_y]
            )
        print(img_adj.shape)
        plt.figure(3)
        plt.imshow(abs(img_adj[1]))
        plt.title('adjoint image')
        plt.show()
        # Test Adjoint property
        x_d = np.vdot(self.multi_img, adj_y)
        x_ad = np.vdot(x, y)
        np.testing.assert_allclose(x_d, x_ad, rtol=1e-5)
        print('done')

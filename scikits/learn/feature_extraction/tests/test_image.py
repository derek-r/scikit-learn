# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD

import numpy as np
import scipy as sp
from scipy import ndimage

from nose.tools import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal

from ..image import img_to_graph
from ..image import extract_patches2d
from ..image import ConvolutionalKMeansEncoder
from ...utils.graph import cs_graph_components

def test_img_to_graph():
    x, y = np.mgrid[:4, :4] - 10
    grad_x = img_to_graph(x)
    grad_y = img_to_graph(y)
    assert_equal(grad_x.nnz, grad_y.nnz)
    # Negative elements are the diagonal: the elements of the original
    # image. Positive elements are the values of the gradient, they
    # shoudl all be equal on grad_x and grad_y
    assert_array_equal(grad_x.data[grad_x.data > 0],
                       grad_y.data[grad_y.data > 0])


def test_connect_regions():
    lena = sp.lena()
    for thr in (50, 150):
        mask = lena > thr
        graph = img_to_graph(lena, mask)
        assert_equal(ndimage.label(mask)[1], cs_graph_components(graph)[0])


def _make_images():
    # Downsample the image by a factor of 4
    lena = sp.lena()
    lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
    lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]

    # make a collection of lenas
    images = np.zeros((3,) + lena.shape)
    images[0] = lena
    images[1] = lena + 1
    images[2] = lena + 2
    return images


def test_extract_patches2d():
    images = _make_images()
    image_size = images.shape[1:]

    # lena is shaped (128, 128): expect 8 * 8 patches with shape (16, 16)
    expected_n_patches = images.shape[0] * 8 * 8

    patches = extract_patches2d(images, image_size, (16, 16))
    assert_equal(patches.shape, (expected_n_patches, 16, 16))

    # extract patches with a 1 offset in the X axis and 3 in the Y axis
    expected_n_patches = images.shape[0] * 7 * 7

    patches = extract_patches2d(images, image_size, (16, 16), offsets=(1, 3))
    assert_equal(patches.shape, (expected_n_patches, 16, 16))


def test_convolutional_kmeans_encoder():
    images = _make_images()
    n_samples = images.shape[0]

    n_centers = 10
    pools = 2

    # expected number of features is driven by number of centers and the number
    # of sum-pooling areas
    n_features = pools * pools * n_centers

    # TODO: make ConvolutionalKMeansEncoder and KMeans rng seedable
    np.random.seed(0)
    encoder = ConvolutionalKMeansEncoder(n_centers=n_centers, n_pools=pools)
    encoder.fit(images)

    assert_almost_equal(encoder.inertia_ / 1e6, .04, 2)
    assert_equal(encoder.filters_.shape, (n_centers, 6 * 6))

    #encoded = encoder.transform(images)
    #assert_equal(encoded.shape, (n_samples, n_features))



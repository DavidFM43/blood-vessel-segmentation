import numpy as np
import pandas as pd

from rle import rle_encode, rle_decode
import torch
import torch.nn as nn
from PIL import Image


# PyTorch version dependence on index data type
torch_ver_major = int(torch.__version__.split('.')[0])
dtype_index = torch.int32 if torch_ver_major >= 2 else torch.long
# _NEIGHBOUR_CODE_TO_NORMALS is a lookup table.
# For every binary neighbour code
# (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes)
# it contains the surface normals of the triangles (called "surfel" for
# "surface element" in the following). The length of the normal
# vector encodes the surfel area.
#
# created using the marching_cube algorithm
# see e.g. https://en.wikipedia.org/wiki/Marching_cubes
# pylint: disable=line-too-long
_NEIGHBOUR_CODE_TO_NORMALS = [[[0, 0, 0]], [[0.125, 0.125, 0.125]],
                              [[-0.125, -0.125, 0.125]],
                              [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
                              [[0.125, -0.125, 0.125]],
                              [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
                              [[0.125, -0.125, 0.125], [-0.125, -0.125,
                                                        0.125]],
                              [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25],
                               [0.125, 0.125, 0.125]], [[-0.125, 0.125,
                                                         0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
                              [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
                              [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125]],
                              [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
                              [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25],
                               [-0.125, 0.125, -0.125]],
                              [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25],
                               [-0.125, 0.125, 0.125]],
                              [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
                              [[0.125, -0.125, -0.125]],
                              [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
                              [[-0.125, -0.125, 0.125],
                               [0.125, -0.125, -0.125]],
                              [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25],
                               [0.125, 0.125, 0.125]],
                              [[0.125, -0.125, 0.125], [0.125, -0.125,
                                                        -0.125]],
                              [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25],
                               [-0.125, -0.125, -0.125]],
                              [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125],
                               [0.125, -0.125, -0.125]],
                              [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25],
                               [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
                              [[-0.125, 0.125, 0.125], [0.125, -0.125,
                                                        -0.125]],
                              [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25],
                               [-0.125, 0.125, 0.125]],
                              [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25],
                               [0.125, -0.125, -0.125]],
                              [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375],
                               [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
                              [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0],
                               [0.25, -0.25, 0.0]],
                              [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25],
                               [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
                              [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125],
                               [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
                              [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125],
                               [-0.25, -0.25, -0.25]], [[0.125, -0.125,
                                                         0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
                              [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125],
                               [0.25, 0.25, -0.25]],
                              [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25],
                               [0.25, 0.0, 0.25]],
                              [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25],
                               [0.125, -0.125, 0.125]],
                              [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25],
                               [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
                              [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
                               [-0.125, 0.125, 0.125]],
                              [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125]],
                              [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25],
                               [0.125, 0.125, -0.125], [-0.125, -0.125,
                                                        0.125]],
                              [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0],
                               [0.25, -0.25, 0.0]],
                              [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25],
                               [-0.125, 0.125, -0.125], [0.125, -0.125,
                                                         0.125]],
                              [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375],
                               [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
                              [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125]],
                              [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
                              [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25],
                               [0.125, -0.125, -0.125]],
                              [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125],
                               [-0.25, 0.25, -0.25]],
                              [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
                              [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0],
                               [0.125, -0.125, 0.125]],
                              [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25],
                               [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
                              [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0],
                               [-0.25, -0.25, -0.25], [-0.125, -0.125,
                                                       -0.125]],
                              [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25],
                               [-0.125, -0.125, -0.125]],
                              [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0],
                               [-0.25, 0.25, 0.0]],
                              [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25],
                               [-0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0],
                               [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
                              [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25],
                               [-0.125, -0.125, 0.125]],
                              [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0],
                               [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
                              [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0],
                               [-0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0],
                               [-0.25, -0.25, 0.0]],
                              [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
                              [[-0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
                              [[-0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0],
                               [0.25, 0.25, -0.0]],
                              [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
                              [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
                               [0.125, -0.125, 0.125]],
                              [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25],
                               [-0.125, -0.125, 0.125]],
                              [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25],
                               [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
                              [[-0.125, -0.125, 0.125], [-0.125, 0.125,
                                                         0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125],
                               [-0.125, 0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25],
                               [-0.25, 0.0, 0.25]],
                              [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25],
                               [0.125, -0.125, 0.125]],
                              [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
                               [-0.125, 0.125, -0.125],
                               [-0.125, 0.125, -0.125]],
                              [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375],
                               [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
                              [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25],
                               [0.125, -0.125, 0.125]],
                              [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
                              [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25],
                               [-0.125, 0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25],
                               [0.25, 0.0, -0.25]],
                              [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375],
                               [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
                              [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25],
                               [-0.125, -0.125, 0.125]],
                              [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
                              [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125],
                               [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
                              [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25],
                               [0.0, 0.0, 0.5]],
                              [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25],
                               [-0.125, 0.125, 0.125]],
                              [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
                               [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25],
                               [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
                              [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25],
                               [0.25, 0.0, 0.25]],
                              [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375],
                               [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
                              [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
                               [0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25],
                               [0.25, 0.0, 0.25]],
                              [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
                              [[-0.125, -0.125, 0.125], [0.125, -0.125,
                                                         0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125],
                               [0.125, -0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25],
                               [0.0, 0.25, -0.25]],
                              [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125],
                               [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
                              [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25],
                               [0.125, -0.125, 0.125]],
                              [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
                               [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25],
                               [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
                              [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25],
                               [0.125, -0.125, -0.125]],
                              [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
                              [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
                               [0.125, -0.125, -0.125]],
                              [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25],
                               [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125],
                               [0.125, -0.125, -0.125]],
                              [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25],
                               [0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
                              [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25],
                               [0.125, -0.125, -0.125]],
                              [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125],
                               [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
                              [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25],
                               [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
                              [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25],
                               [-0.125, 0.125, 0.125]],
                              [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0],
                               [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
                              [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25],
                               [0.0, 0.0, 0.5]],
                              [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25],
                               [0.0, 0.25, 0.25]],
                              [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
                              [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25],
                               [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
                              [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125],
                               [0.125, 0.125, 0.125]],
                              [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25],
                               [0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0],
                               [0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125]], [[0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
                              [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0],
                               [0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25],
                               [0.125, 0.125, 0.125]],
                              [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125],
                               [0.125, 0.125, 0.125]],
                              [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25],
                               [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
                              [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
                              [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25],
                               [0.0, 0.25, 0.25]],
                              [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25],
                               [0.0, 0.0, 0.5]],
                              [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0],
                               [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
                              [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25],
                               [-0.125, 0.125, 0.125]],
                              [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25],
                               [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
                              [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125],
                               [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
                              [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25],
                               [0.125, -0.125, -0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
                              [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25],
                               [0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125],
                               [0.125, -0.125, -0.125]],
                              [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25],
                               [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
                               [0.125, -0.125, -0.125]],
                              [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
                              [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25],
                               [0.125, -0.125, -0.125]],
                              [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25],
                               [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
                              [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
                               [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25],
                               [0.125, -0.125, 0.125]],
                              [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125],
                               [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25],
                               [0.0, 0.25, -0.25]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125],
                               [0.125, -0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [0.125, -0.125,
                                                         0.125]],
                              [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
                              [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25],
                               [0.25, 0.0, 0.25]],
                              [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
                               [0.125, -0.125, 0.125]],
                              [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375],
                               [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
                              [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25],
                               [0.25, 0.0, 0.25]],
                              [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25],
                               [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
                              [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
                               [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25],
                               [-0.125, 0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25],
                               [0.0, 0.0, 0.5]],
                              [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125],
                               [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
                              [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
                              [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25],
                               [-0.125, -0.125, 0.125]],
                              [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375],
                               [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25],
                               [0.25, 0.0, -0.25]],
                              [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25],
                               [-0.125, 0.125, 0.125]],
                              [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
                              [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25],
                               [0.125, -0.125, 0.125]],
                              [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375],
                               [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
                              [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
                               [-0.125, 0.125, -0.125],
                               [-0.125, 0.125, -0.125]],
                              [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25],
                               [0.125, -0.125, 0.125]],
                              [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25],
                               [-0.25, 0.0, 0.25]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125],
                               [-0.125, 0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [-0.125, 0.125,
                                                         0.125]],
                              [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25],
                               [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
                              [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25],
                               [-0.125, -0.125, 0.125]],
                              [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25],
                               [0.125, -0.125, 0.125]],
                              [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
                              [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0],
                               [0.25, 0.25, -0.0]],
                              [[-0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
                              [[-0.125, -0.125, 0.125]],
                              [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
                              [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0],
                               [-0.25, -0.25, 0.0]],
                              [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0],
                               [-0.125, -0.125, 0.125]],
                              [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0],
                               [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
                              [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25],
                               [-0.125, -0.125, 0.125]],
                              [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0],
                               [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
                              [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25],
                               [-0.125, -0.125, 0.125],
                               [-0.125, -0.125, 0.125]],
                              [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0],
                               [-0.25, 0.25, 0.0]],
                              [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25],
                               [-0.125, -0.125, -0.125]],
                              [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0],
                               [-0.25, -0.25, -0.25], [-0.125, -0.125,
                                                       -0.125]],
                              [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25],
                               [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
                              [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0],
                               [0.125, -0.125, 0.125]],
                              [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
                              [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125],
                               [-0.25, 0.25, -0.25]],
                              [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25],
                               [0.125, -0.125, -0.125]],
                              [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
                              [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125]],
                              [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375],
                               [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
                              [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25],
                               [-0.125, 0.125, -0.125], [0.125, -0.125,
                                                         0.125]],
                              [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0],
                               [0.25, -0.25, 0.0]],
                              [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25],
                               [0.125, 0.125, -0.125], [-0.125, -0.125,
                                                        0.125]],
                              [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125],
                               [-0.125, 0.125, 0.125]],
                              [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25],
                               [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
                              [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25],
                               [0.125, -0.125, 0.125]],
                              [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25],
                               [0.25, 0.0, 0.25]],
                              [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125],
                               [0.25, 0.25, -0.25]],
                              [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
                              [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
                              [[0.125, -0.125, 0.125]],
                              [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125],
                               [-0.25, -0.25, -0.25]],
                              [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125],
                               [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
                              [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25],
                               [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
                              [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0],
                               [0.25, -0.25, 0.0]],
                              [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375],
                               [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
                              [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25],
                               [0.125, -0.125, -0.125]],
                              [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25],
                               [-0.125, 0.125, 0.125]],
                              [[-0.125, 0.125, 0.125], [0.125, -0.125,
                                                        -0.125]],
                              [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25],
                               [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
                              [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125],
                               [0.125, -0.125, -0.125]],
                              [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25],
                               [-0.125, -0.125, -0.125]],
                              [[0.125, -0.125, 0.125], [0.125, -0.125,
                                                        -0.125]],
                              [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25],
                               [0.125, 0.125, 0.125]],
                              [[-0.125, -0.125, 0.125],
                               [0.125, -0.125, -0.125]],
                              [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
                              [[0.125, -0.125, -0.125]],
                              [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
                              [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25],
                               [-0.125, 0.125, 0.125]],
                              [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25],
                               [-0.125, 0.125, -0.125]],
                              [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
                              [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25],
                               [-0.125, -0.125, 0.125]],
                              [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
                              [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
                              [[-0.125, 0.125, 0.125]],
                              [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25],
                               [0.125, 0.125, 0.125]],
                              [[0.125, -0.125, 0.125], [-0.125, -0.125,
                                                        0.125]],
                              [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
                              [[0.125, -0.125, 0.125]],
                              [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
                              [[-0.125, -0.125, 0.125]],
                              [[0.125, 0.125, 0.125]], [[0, 0, 0]]]


class SurfaceDiceMetric:
    def __init__(self, n_batches, device):
        self.n_batches = n_batches
        self.device = device
        self.area = torch.from_numpy(self.create_table_neighbour_code_to_surface_area((1, 1, 1))).to(self.device)
        self.unfold = torch.nn.Unfold(kernel_size=(2, 2), padding=1)
        self.batch_idx = 0
        self.numerator = 0
        self.denominator = 0
        self.pred_pad = -1
        self.target_pad = -1

    def create_table_neighbour_code_to_surface_area(self, spacing_mm):
        neighbour_code_to_surface_area = np.zeros([256])
        for code in range(256):
            normals = np.array(_NEIGHBOUR_CODE_TO_NORMALS[code])
            sum_area = 0
            for normal_idx in range(normals.shape[0]):
                # normal vector
                n = np.zeros([3])
                n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
                n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
                n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
                area = np.linalg.norm(n)
                sum_area += area
            neighbour_code_to_surface_area[code] = sum_area

        print(neighbour_code_to_surface_area.dtype)
        return neighbour_code_to_surface_area

    def compute_surface_area(self, surface):
        d, h, w = surface.shape
        surface = surface.view(1, 1, d, h, w).float()
        weight = (2**torch.arange(8, device=self.device)).view(1, 1, 2, 2, 2).float()
        cubes_byte = torch.nn.functional.conv3d(surface, weight, padding=(0, 1, 1)).flatten()

        cubes_area = self.area[cubes_byte.int()]
        return cubes_area

    def process_batch(self, pred, target):
        bs, h, w = pred.shape
        pad_d = int(bs % 2 == 0) + 1
        pad = torch.zeros((pad_d, h, w), dtype=torch.uint8, device=self.device)

        if self.batch_idx == 0:
            pred = torch.vstack([pad, pred])
            target = torch.vstack([pad, target])
        elif self.batch_idx == self.n_batches - 1:
            pred = torch.vstack([pred, pad])
            target = torch.vstack([target, pad])
        else:
            pred = torch.vstack([self.pred_pad, pred])
            target = torch.vstack([self.target_pad, target])

        area_pred_total = self.compute_surface_area(pred)
        area_true_total = self.compute_surface_area(target)
        idx = torch.logical_and(area_pred_total > 0, area_true_total > 0)
        n1 = area_pred_total[idx].sum().item() + area_true_total[idx].sum().item()
        d1 = area_pred_total.sum().item() + area_true_total.sum().item()
        # n1 = area_pred_total[idx].view(k, -1).sum(1).sum() + area_true_total[idx].view(k, -1).sum(1).sum() 
        # d1 = area_pred_total.view(k, -1).sum(1).sum() + area_true_total.view(k, -1).sum(1).sum()



        area_pred_total_2 = torch.tensor([])
        area_true_total_2 = torch.tensor([])
        n2 = 0
        d2 = 0

        for window in range(len(pred) - 1):
            area_pred = self.compute_surface_area(pred[window:window+2])
            area_true = self.compute_surface_area(target[window:window+2])
            
            area_pred_total_2 = torch.cat([area_pred_total_2, area_pred])
            area_true_total_2 = torch.cat([area_true_total_2, area_true])

            idx = torch.logical_and(area_pred > 0, area_true > 0)

            self.numerator += area_pred[idx].sum() + area_true[idx].sum()
            self.denominator += area_pred.sum() + area_true.sum()

            n2 += area_pred[idx].sum().item() + area_true[idx].sum().item()
            d2 += area_pred.sum().item() + area_true.sum().item()
        

        # idx2 = torch.logical_and(area_pred_total_2 > 0, area_true_total_2 > 0)
        # n2 = area_pred_total_2[idx2].sum().item() + area_true_total_2[idx2].sum().item()
        # d2 = area_pred_total_2.sum().item() + area_true_total_2.sum().item()

        if n1 == n2:
            print(True)
        else:
            print("numerator", abs(n1-n2))
        if d1 == d2:
            print(True)
        else:
            print("denominator", abs(d1- d2))

        self.batch_idx += 1
        self.pred_pad = pred[-1:]
        self.target_pad = target[-1:]

    def compute_metric(self):
        dice = self.numerator / self.denominator.clamp(min=1e-8)
        return dice.item()
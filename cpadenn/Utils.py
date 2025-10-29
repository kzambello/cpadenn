# Copyright 2025 Kevin Zambello
#
# This program is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

"""Module providing misc utilities."""

import numpy as np
from scipy import ndimage

import tensorflow as tf

import cpadenn


def FindLocMax(Xg, Yg, obs, size=16):
    """Find local maximums for observable obs."""

    locmax = ndimage.maximum_filter(obs, size)

    mymask = obs == locmax

    res = np.stack([Xg[mymask].ravel(), Yg[mymask].ravel()], axis=-1).astype(np.float32)

    return res


# zrelu, https://arxiv.org/abs/1602.09046
def zrelu(z):
    """Complex ReLU: zrelu"""

    z = z[:, 0] + 1.0j * z[:, 1]
    theta = np.angle(z)
    mybool = (theta > 0) & (theta < np.pi / 2.0)
    res = z * mybool
    return np.stack([res.real, res.imag], axis=-1)


# complex cardioid, https://arxiv.org/abs/1707.00070
def ccardioid(z):
    """Complex ReLU: complex cardioid"""

    z = z[:, 0] + 1.0j * z[:, 1]
    res = (1.0 + np.cos(np.angle(z))) * z / 2.0
    return np.stack([res.real, res.imag], axis=-1)


# modrelu, https://arxiv.org/pdf/1511.06464
def modrelu(z, b=0.0):
    """Complex ReLU: modrelu"""

    z = z[:, 0] + 1.0j * z[:, 1]
    norm = np.abs(z)
    mybool = (norm + b) >= 0.0
    res = (norm + b) * z / norm * mybool
    return np.stack([res.real, res.imag], axis=-1)


# Exempli gratia:
#
# def crelu(z):
#    return cpadenn.Utils.ccardioid(z)
#
# alphas11 = cpadenn.Utils.CPadeAF_calibration(1, 1, crelu)
# alphas_safe11 = cpadenn.Utils.CPadeAF_calibration(1, 1, crelu, safe=True)
#
# alphas22 = cpadenn.Utils.CPadeAF_calibration(2, 2, crelu)
# alphas_safe22 = cpadenn.Utils.CPadeAF_calibration(2, 2, crelu, safe=True)
#
# alphas33 = cpadenn.Utils.CPadeAF_calibration(3, 3, crelu)
# alphas_safe33 = cpadenn.Utils.CPadeAF_calibration(3, 3, crelu, safe=True)
#
# alphas44 = cpadenn.Utils.CPadeAF_calibration(4, 4, crelu)
# alphas_safe44 = cpadenn.Utils.CPadeAF_calibration(4, 4, crelu, safe=True)


def CPadeAF_calibration(deg_num, deg_den, crelu, safe=False):
    """Compute initial weights for Complex Pade activation function."""

    inputs = tf.keras.layers.Input(shape=(2,))

    x = inputs
    x = cpadenn.Layers.CMergeReIm()(x)
    x = cpadenn.Layers.CPadeAF(
        deg_num=deg_num,
        deg_den=deg_den,
        lreg=1.0e-3,
        imean=0.0,
        istddev=1.0e-4,
        safe=safe,
    )(x)
    x = cpadenn.Layers.CSplitReIm()(x)

    outputs = x

    mymodel = tf.keras.Model(inputs=inputs, outputs=outputs)
    optadam = tf.keras.optimizers.Adam(learning_rate=1.0e-3)

    mymodel.compile(optimizer=optadam, loss="mean_squared_error")

    pts = 500
    xg = np.linspace(-3, 3, pts)
    yg = np.linspace(-3, 3, pts)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = np.stack([Xg.ravel(), Yg.ravel()], axis=-1).astype(np.float32)

    print("\nCalibrating (1)...\n")
    mymodel.fit(Zg, crelu(Zg), epochs=100, batch_size=2048)

    print("\nCalibrating (2)...\n")
    mymodel.fit(Zg, crelu(Zg), epochs=100, batch_size=2048)

    alphas = mymodel.get_weights()

    print("\nWeights:\n\n", alphas, "\n")

    return alphas


def get_alphas(n, safe=False):
    """
    Return precomputed initial weights for Complex Pade activation function of order n/n.
    """

    if safe is True:
        print(
            "WARNING: no precomputed alphas for safe Complex Pade activation functions, returning "
            "None."
        )
        return None

    if n == 1:
        alphas = [
            np.array([0.44639155, 0.5874375], dtype=np.float32),
            np.array([-0.11688029], dtype=np.float32),
            np.array([-0.00164319, 0.00171649], dtype=np.float32),
            np.array([0.00100579], dtype=np.float32),
        ]
    elif n == 2:
        alphas = [
            np.array([0.4516767, 0.05266491, 0.5457554], dtype=np.float32),
            np.array([-0.08004141, 0.01517201], dtype=np.float32),
            np.array([-0.00038721, -0.00039545, -0.0014949], dtype=np.float32),
            np.array([-0.00040207, -0.00024268], dtype=np.float32),
        ]
    elif n == 3:
        alphas = [
            np.array([0.38862282, 0.00826987, 0.00152397, 0.5423925], dtype=np.float32),
            np.array([-0.20875129, 0.0437681, -0.00415791], dtype=np.float32),
            np.array(
                [0.00055318, -0.00013447, -0.00018349, 0.00097038], dtype=np.float32
            ),
            np.array([0.00019213, 0.00013342, -0.00011132], dtype=np.float32),
        ]
    elif n == 4:
        alphas = [
            np.array(
                [0.54397285, 0.1292388, 0.01105397, 0.00199061, 0.54465675],
                dtype=np.float32,
            ),
            np.array(
                [0.08190799, -0.00250178, 0.00801145, -0.00114474], dtype=np.float32
            ),
            np.array(
                [-0.02204097, -0.01346053, 0.00094724, 0.00035264, 0.00064983],
                dtype=np.float32,
            ),
            np.array(
                [-3.8732085e-02, 9.0270257e-03, -1.7697400e-03, 9.7313503e-05],
                dtype=np.float32,
            ),
        ]
    else:
        print(f"WARNING: no precomputed alphas for n = {n}, returning None.")
        return None

    return alphas

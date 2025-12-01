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


def get_alphas(n, m, safe=False):
    """
    Return precomputed initial weights for Complex Pade activation function of order n/n.
    """

    if safe is True:
        if n == 3 and m == 3:
            alphas = [
                np.array(
                    [0.7028826, 0.09448481, -0.00894883, 0.4818811], dtype=np.float32
                ),
                np.array([-0.09537569, 0.04550191, -0.00649218], dtype=np.float32),
                np.array(
                    [-0.00111217, 0.00048354, 0.00010253, 0.0013575], dtype=np.float32
                ),
                np.array([-0.1694703, 0.07872267, -0.01214926], dtype=np.float32),
            ]
        elif n == 3 and m == 2:
            alphas = [
                np.array(
                    [0.57864904, 0.09365933, -0.00282509, 0.5841132], dtype=np.float32
                ),
                np.array([0.00041374, 0.00031814], dtype=np.float32),
                np.array(
                    [1.1065153e-03, -7.6211587e-04, -5.3236025e-05, -1.5721197e-04],
                    dtype=np.float32,
                ),
                np.array([0.0626029, -0.01708659], dtype=np.float32),
            ]
        elif n == 4 and m == 4:
            alphas = [
                np.array(
                    [0.7257368, 0.08196919, -0.01256165, 0.00140988, 0.37371516],
                    dtype=np.float32,
                ),
                np.array(
                    [0.16024841, -0.11743093, 0.0327547, -0.00313357], dtype=np.float32
                ),
                np.array(
                    [
                        -1.2159096e-03,
                        8.8814850e-04,
                        1.7775426e-04,
                        -8.8957997e-05,
                        1.4857965e-03,
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [-0.11850382, 0.08703844, -0.02366448, 0.0029188], dtype=np.float32
                ),
            ]
        else:
            print(
                f"WARNING: no precomputed alphas for (n,m) = ({n},{m}) returning None."
            )
            return None

        return alphas

    if n == 1 and m == 1:
        alphas = [
            np.array([0.44639155, 0.5874375], dtype=np.float32),
            np.array([-0.11688029], dtype=np.float32),
            np.array([-0.00164319, 0.00171649], dtype=np.float32),
            np.array([0.00100579], dtype=np.float32),
        ]
    elif n == 2 and m == 2:
        alphas = [
            np.array([0.4516767, 0.05266491, 0.5457554], dtype=np.float32),
            np.array([-0.08004141, 0.01517201], dtype=np.float32),
            np.array([-0.00038721, -0.00039545, -0.0014949], dtype=np.float32),
            np.array([-0.00040207, -0.00024268], dtype=np.float32),
        ]
    elif n == 3 and m == 3:
        alphas = [
            np.array([0.38862282, 0.00826987, 0.00152397, 0.5423925], dtype=np.float32),
            np.array([-0.20875129, 0.0437681, -0.00415791], dtype=np.float32),
            np.array(
                [0.00055318, -0.00013447, -0.00018349, 0.00097038], dtype=np.float32
            ),
            np.array([0.00019213, 0.00013342, -0.00011132], dtype=np.float32),
        ]
    elif n == 3 and m == 2:
        alphas = [
            np.array([0.49811104, 0.09632787, 0.00930952, 0.5379451], dtype=np.float32),
            np.array([-0.00268338, 0.0204672], dtype=np.float32),
            np.array(
                [0.00047284, -0.00017495, 0.00052347, 0.00051169], dtype=np.float32
            ),
            np.array([2.4028297e-05, -1.4469674e-04], dtype=np.float32),
        ]
    elif n == 4 and m == 4:
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
        print(f"WARNING: no precomputed alphas for (n,m) = ({n},{m}) returning None.")
        return None

    return alphas

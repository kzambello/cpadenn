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

import PadeNN

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
#    return PadeNN.Utils.ccardioid(z)
#
# alphas11 = PadeNN.Utils.CPadeAF_calibration(1, 1, crelu)
# alphas_safe11 = PadeNN.Utils.CPadeAF_calibration(1, 1, crelu, safe=True)
#
# alphas22 = PadeNN.Utils.CPadeAF_calibration(2, 2, crelu)
# alphas_safe22 = PadeNN.Utils.CPadeAF_calibration(2, 2, crelu, safe=True)
#
# alphas33 = PadeNN.Utils.CPadeAF_calibration(3, 3, crelu)
# alphas_safe33 = PadeNN.Utils.CPadeAF_calibration(3, 3, crelu, safe=True)
#
# alphas44 = PadeNN.Utils.CPadeAF_calibration(4, 4, crelu)
# alphas_safe44 = PadeNN.Utils.CPadeAF_calibration(4, 4, crelu, safe=True)


def CPadeAF_calibration(deg_num, deg_den, crelu, safe=False):
    """Compute initial weights for Complex Pade activation function."""

    inputs = tf.keras.layers.Input(shape=(2,))

    x = inputs
    x = PadeNN.Layers.CMergeReIm()(x)
    x = PadeNN.Layers.CPadeAF(
        deg_num=deg_num,
        deg_den=deg_den,
        lreg=1.0e-3,
        imean=0.0,
        istddev=1.0e-4,
        safe=safe,
    )(x)
    x = PadeNN.Layers.CSplitReIm()(x)

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


def get_alphas(n):
    """Return precomputed initial weights for (unsafe) Complex Pade activation
       function of order n/n.
    """

    if n == 4:
        alphas = [
            np.array(
                [0.55991364, 0.14828739, 0.01649525, 0.00223316, 0.5429329],
                dtype=np.float32,
            ),
            np.array(
                [0.11221588, 0.00310184, 0.00828541, -0.0009095], dtype=np.float32
            ),
            np.array(
                [
                    -6.2417763e-04,
                    -6.3668203e-04,
                    8.2182858e-05,
                    1.6188879e-04,
                    -5.9690198e-04,
                ],
                dtype=np.float32,
            ),
            np.array(
                [-1.0941960e-03, 2.2226348e-05, -2.5839164e-04, -1.7531947e-04],
                dtype=np.float32,
            ),
        ]
        return alphas

    print(f"WARNING: no precomputed alphas for n = {n}, returning None.")
    return None

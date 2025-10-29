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

"""Module providing building blocks for a Pade neural network."""

import tensorflow as tf


class CDense(tf.keras.layers.Layer):
    """Complex dense layer."""

    def __init__(self, units=4, lreg=1.0e-4, imean=0.0, istddev=1.0):
        super().__init__()

        self.units = units
        self.lreg = lreg

        self.imean = imean
        self.istddev = istddev

    def build(self, input_shape):
        """Build method."""

        feature_dim = input_shape[-1]
        # if input_shape is (batch_size,), input_shape[-1] is batch_size but feature_dim is 1
        if len(input_shape) == 1:
            feature_dim = 1

        initializer = tf.keras.initializers.TruncatedNormal(
            mean=self.imean, stddev=self.istddev
        )
        regularizer = tf.keras.regularizers.l2(self.lreg)

        self.w_r = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )
        self.b_r = self.add_weight(
            shape=(self.units,),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )

        self.w_i = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )
        self.b_i = self.add_weight(
            shape=(self.units,),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )

    def call(self, inputs):
        """Call method."""

        inputs_c = inputs[:]

        # if input_shape is (batch_size,), reshape to (batch_size, 1)
        if len(inputs_c.shape) == 1:
            inputs_c = tf.expand_dims(inputs_c, axis=1)

        w_c = tf.complex(self.w_r[:], self.w_i[:])
        b_c = tf.complex(self.b_r[:], self.b_i[:])

        inputs_c = tf.cast(inputs_c, tf.complex128)
        w_c = tf.cast(w_c, tf.complex128)
        b_c = tf.cast(b_c, tf.complex128)

        batch_size = tf.shape(inputs_c)[0]
        w_c_batched = tf.broadcast_to(w_c, [batch_size, w_c.shape[0], w_c.shape[1]])
        inputs_c = tf.expand_dims(
            inputs_c, axis=1
        )  # shape becomes (batch_size, 1, feature_dim)
        res = tf.matmul(inputs_c, w_c_batched)
        res = tf.squeeze(res, axis=1) + b_c

        return res


class CFugacityCoV(tf.keras.layers.Layer):
    """Fugacity change of variables."""

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """Call method."""

        inputs_c = tf.complex(inputs[:, 0], inputs[:, 1])

        inputs_c = tf.cast(inputs_c, tf.complex128)

        res = tf.exp(inputs_c)
        res = tf.reshape(
            res, (-1, 1)
        )  # reshape output from (batch_size,) to (batch_size, 1)

        return res


class CPadeAF(tf.keras.layers.Layer):
    """Complex Pade activation function."""

    def __init__(
        self,
        deg_num=3,
        deg_den=3,
        lreg=1.0e-4,
        alphas=None,
        imean=0.0,
        istddev=1.0,
        safe=False,
    ):
        super().__init__()

        # The variable safe_kind is hard-coded, for now. Available options are:
        #
        #    SPAU:  complexified version of https://arxiv.org/pdf/1907.06732
        #    ERA:   complexified version of https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800705.pdf
        #    modERA: ERA modified to not have zeros on imaginary (instead of real) axis
        #

        self.safe_kind = "SPAU"

        self.deg_num = deg_num
        self.deg_den = deg_den

        if self.safe_kind in ('ERA', 'modERA') and safe is True:
            new_deg_den = deg_den / 2.0
            new_deg_den = int(2 * round(new_deg_den / 2))  # closest even int
            self.deg_den = new_deg_den
            print(
                f"Warning: using experimental {self.safe_kind} rational function, adjusting internal deg_den from {deg_den} to {new_deg_den}. Effective degree is {2*new_deg_den}."
            )

        self.lreg = lreg
        self.alphas = alphas
        self.imean = imean
        self.istddev = istddev
        self.safe = safe

        initializer = tf.keras.initializers.TruncatedNormal(mean=imean, stddev=istddev)
        regularizer = tf.keras.regularizers.l2(self.lreg)

        self.coeffs_num_r = self.add_weight(
            shape=(deg_num + 1,),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )
        self.coeffs_den_r = self.add_weight(
            shape=(deg_den,),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )
        self.coeffs_num_i = self.add_weight(
            shape=(deg_num + 1,),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )
        self.coeffs_den_i = self.add_weight(
            shape=(deg_den,),
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,
        )

        if self.alphas is not None:
            self.coeffs_num_r.assign(self.alphas[0])
            self.coeffs_den_r.assign(self.alphas[1])
            self.coeffs_num_i.assign(self.alphas[2])
            self.coeffs_den_i.assign(self.alphas[3])

    def call(self, inputs):
        """Call method."""

        inputs_c = inputs[:]

        coeffs_num_c = tf.complex(self.coeffs_num_r[:], self.coeffs_num_i[:])
        coeffs_den_c = tf.complex(self.coeffs_den_r[:], self.coeffs_den_i[:])

        inputs_c = tf.cast(inputs_c, tf.complex128)
        coeffs_num_c = tf.cast(coeffs_num_c, tf.complex128)
        coeffs_den_c = tf.cast(coeffs_den_c, tf.complex128)
        one_c = tf.cast(tf.complex(1.0, 0.0), tf.complex128)
        onej_c = tf.cast(tf.complex(0.0, 1.0), tf.complex128)
        epsilon_c = tf.cast(tf.complex(1.0e-6, 0.0), tf.complex128)

        numerator = tf.add_n(
            [coeffs_num_c[i] * tf.pow(inputs_c, i + 1) for i in range(self.deg_num)]
        )
        numerator = tf.add(numerator, coeffs_num_c[self.deg_num])

        denominator = tf.add_n(
            [coeffs_den_c[i] * tf.pow(inputs_c, i + 1) for i in range(self.deg_den)]
        )

        if self.safe is True and self.safe_kind == "SPAU":
            absden = tf.sqrt(denominator * tf.math.conj(denominator))
            denominator = tf.cast(absden, tf.complex128) + one_c
        else:
            denominator = denominator + one_c

        if self.safe is True and self.safe_kind == "ERA":
            denominator = one_c
            for i in range(self.deg_den):
                # term = (z+c)^2 + d^2
                c = tf.cast(
                    tf.complex(
                        tf.math.real(coeffs_den_c[i]),
                        tf.constant(0.0, dtype=tf.float64),
                    ),
                    tf.complex128,
                )
                d = tf.cast(
                    tf.complex(
                        tf.math.imag(coeffs_den_c[i]),
                        tf.constant(0.0, dtype=tf.float64),
                    ),
                    tf.complex128,
                )
                term = tf.pow(inputs_c - c, 2) + tf.pow(d, 2)
                denominator = tf.multiply(denominator, term)

            denominator = denominator + epsilon_c

        if self.safe is True and self.safe_kind == "modERA":
            denominator = one_c
            for i in range(self.deg_den):
                # term = -(z+ic)^2 + d^2
                c = tf.cast(
                    tf.complex(
                        tf.math.real(coeffs_den_c[i]),
                        tf.constant(0.0, dtype=tf.float64),
                    ),
                    tf.complex128,
                )
                d = tf.cast(
                    tf.complex(
                        tf.math.imag(coeffs_den_c[i]),
                        tf.constant(0.0, dtype=tf.float64),
                    ),
                    tf.complex128,
                )
                term = -tf.pow(inputs_c - onej_c * c, 2) + tf.pow(d, 2)
                denominator = tf.multiply(denominator, term)

            denominator = denominator + epsilon_c

        res = tf.math.divide_no_nan(numerator, denominator)

        return res


# complex cardioid, https://arxiv.org/abs/1707.00070
class CReLUAF_ccardioid(tf.keras.layers.Layer):
    """Complex ReLU activation function (based on complex cardioid)."""

    def __init__(self):
        super().__init__()

    def call(self, x):
        """Call method."""

        one_c = tf.constant(1.0, dtype=tf.complex128)
        two_c = tf.constant(2.0, dtype=tf.complex128)
        xangle = tf.math.angle(x)
        xangle = tf.complex(xangle, tf.zeros_like(xangle))
        res = one_c + tf.cast(tf.math.cos(xangle), dtype=tf.complex128) * x / two_c
        return res


# zrelu, https://arxiv.org/abs/1602.09046
class CReLUAF_zrelu(tf.keras.layers.Layer):
    """Complex ReLU activation function (based on zrelu)."""

    def __init__(self):
        super().__init__()

    def call(self, x):
        """Call method."""

        xangle = tf.math.angle(x)
        pi_half = tf.constant(3.141592653589793 / 2.0, dtype=tf.float64)
        mybool = tf.logical_and(xangle > 0.0, xangle < pi_half)
        mybool = tf.cast(mybool, dtype=tf.complex128)
        res = mybool * x
        return res


# alias
CReLUAF = CReLUAF_ccardioid


class CSplitReIm(tf.keras.layers.Layer):
    """Split real and imaginary parts."""

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """Call method."""

        inputs_c = inputs[:]

        res = tf.cast(inputs_c, tf.complex128)

        res = tf.squeeze(res, axis=1)

        return tf.stack([tf.math.real(res), tf.math.imag(res)], axis=-1)


class CMergeReIm(tf.keras.layers.Layer):
    """Merge real and imaginary parts."""

    def __init__(self):
        super().__init__()

    def call(self, inputs):
        """Call method."""

        inputs_c = tf.complex(inputs[:, 0], inputs[:, 1])

        inputs_c = tf.cast(inputs_c, tf.complex128)

        res = inputs_c

        res = tf.reshape(
            res, (-1, 1)
        )  # reshape output from (batch_size,) to  (batch_size, 1)

        return res


class CDropout(tf.keras.layers.Layer):
    """Complex dropout layer."""

    def __init__(self, rate=0.1):
        super().__init__()
        self.rate = rate
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        """Call method."""

        inputs_r = tf.math.real(inputs)
        inputs_i = tf.math.imag(inputs)

        return tf.complex(
            self.dropout(inputs_r, training=training),
            self.dropout(inputs_i, training=training),
        )

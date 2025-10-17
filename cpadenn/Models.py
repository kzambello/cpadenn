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

"""Module providing models for a Pade neural network."""

import tensorflow as tf

from cpadenn import Layers, Utils


class PadeModel(tf.keras.Model):
    """Complex Pade model."""

    def __init__(self, lreg=1.0e-3, safe=False, fugacity=True):
        super().__init__()

        if fugacity:
            self.cmergereim = Layers.CFugacityCoV()
        else:
            self.cmergereim = Layers.CMergeReIm()

        self.cdense1 = Layers.CDense(units=8, lreg=lreg)
        self.cpadeaf1 = Layers.CPadeAF(
            deg_num=4, deg_den=4, lreg=lreg, alphas=Utils.get_alphas(4), safe=safe
        )

        self.cdense2 = Layers.CDense(units=4, lreg=lreg)
        self.cpadeaf2 = Layers.CPadeAF(
            deg_num=4, deg_den=4, lreg=lreg, alphas=Utils.get_alphas(4), safe=safe
        )

        self.cdense3 = Layers.CDense(units=1, lreg=lreg)
        self.csplitreim = Layers.CSplitReIm()

    def call(self, inputs):
        """Call method."""

        x = self.cmergereim(inputs)
        x = self.cdense1(x)
        x = self.cpadeaf1(x)
        x = self.cdense2(x)
        x = self.cpadeaf2(x)
        x = self.cdense3(x)
        x = self.csplitreim(x)

        return x


class BaselineModel1(tf.keras.Model):
    """Complex ReLU model."""

    def __init__(self, lreg=1.0e-3, fugacity=True):
        super().__init__()

        if fugacity:
            self.cmergereim = Layers.CFugacityCoV()
        else:
            self.cmergereim = Layers.CMergeReIm()

        self.cdense1 = Layers.CDense(units=8, lreg=lreg)
        self.creluaf1 = Layers.CReLUAF()

        self.cdense2 = Layers.CDense(units=4, lreg=lreg)
        self.creluaf2 = Layers.CReLUAF()

        self.cdense3 = Layers.CDense(units=1, lreg=lreg)
        self.csplitreim = Layers.CSplitReIm()

    def call(self, inputs):
        """Call method."""

        x = self.cmergereim(inputs)
        x = self.cdense1(x)
        x = self.creluaf1(x)
        x = self.cdense2(x)
        x = self.creluaf2(x)
        x = self.cdense3(x)
        x = self.csplitreim(x)

        return x


class BaselineModel2(tf.keras.Model):
    """
    Complex ReLU model with approximately same number of trainable parameters
    as Complex Pade model.
    """

    def __init__(self, lreg=1.0e-3, fugacity=True):
        super().__init__()

        if fugacity:
            self.cmergereim = Layers.CFugacityCoV()
        else:
            self.cmergereim = Layers.CMergeReIm()

        self.cdense1 = Layers.CDense(units=8, lreg=lreg)
        self.creluaf1 = Layers.CReLUAF()

        self.cdense2 = Layers.CDense(units=6, lreg=lreg)
        self.creluaf2 = Layers.CReLUAF()

        self.cdense3 = Layers.CDense(units=1, lreg=lreg)
        self.csplitreim = Layers.CSplitReIm()

    def call(self, inputs):
        """Call method."""

        x = self.cmergereim(inputs)
        x = self.cdense1(x)
        x = self.creluaf1(x)
        x = self.cdense2(x)
        x = self.creluaf2(x)
        x = self.cdense3(x)
        x = self.csplitreim(x)

        return x


class CustomModel(tf.keras.Model):
    """Complex custom model."""

    def __init__(self, lreg=1.0e-3, fugacity=True, units=[8, 4], n=4, safe=False):
        super().__init__()

        if fugacity:
            self.cmergereim = Layers.CFugacityCoV()
        else:
            self.cmergereim = Layers.CMergeReIm()

        self.hidden_layers = []

        for u in units:
            self.hidden_layers.append(Layers.CDense(units=u, lreg=lreg))

            if n == 0:
                self.hidden_layers.append(Layers.CReLUAF())
            else:
                if n > 4:
                    n = 4
                    print(
                        f"WARNING: requested Pade activation function of order n = {n} > 4, overriding to n = 4."
                    )
                self.hidden_layers.append(
                    Layers.CPadeAF(
                        deg_num=n,
                        deg_den=n,
                        lreg=lreg,
                        alphas=Utils.get_alphas(n),
                        safe=safe,
                    )
                )

        self.cdense = Layers.CDense(units=1, lreg=lreg)
        self.csplitreim = Layers.CSplitReIm()

    def call(self, inputs):
        """Call method."""

        x = self.cmergereim(inputs)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.cdense(x)
        x = self.csplitreim(x)

        return x

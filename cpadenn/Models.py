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

from PadeNN import Layers, Utils


class PadeModel(tf.keras.Model):
    """Complex Pade model."""

    def __init__(self, lreg=1.0e-3, safe=False):
        super().__init__()

        self.cfugacitycov = Layers.CFugacityCoV()

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

        x = self.cfugacitycov(inputs)
        x = self.cdense1(x)
        x = self.cpadeaf1(x)
        x = self.cdense2(x)
        x = self.cpadeaf2(x)
        x = self.cdense3(x)
        x = self.csplitreim(x)

        return x


class Baseline1Model(tf.keras.Model):
    """Complex ReLU model."""

    def __init__(self, lreg=1.0e-3):
        super().__init__()

        self.cfugacitycov = Layers.CFugacityCoV()

        self.cdense1 = Layers.CDense(units=8, lreg=lreg)
        self.creluaf1 = Layers.CReLUAF()

        self.cdense2 = Layers.CDense(units=4, lreg=lreg)
        self.creluaf2 = Layers.CReLUAF()

        self.cdense3 = Layers.CDense(units=1, lreg=lreg)
        self.csplitreim = Layers.CSplitReIm()

    def call(self, inputs):
        """Call method."""

        x = self.cfugacitycov(inputs)
        x = self.cdense1(x)
        x = self.creluaf1(x)
        x = self.cdense2(x)
        x = self.creluaf2(x)
        x = self.cdense3(x)
        x = self.csplitreim(x)

        return x


class Baseline2Model(tf.keras.Model):
    """
       Complex ReLU model with approximately same number of trainable parameters
       as Complex Pade model.
    """

    def __init__(self, lreg=1.0e-3):
        super().__init__()

        self.cfugacitycov = Layers.CFugacityCoV()

        self.cdense1 = Layers.CDense(units=8, lreg=lreg)
        self.creluaf1 = Layers.CReLUAF()

        self.cdense2 = Layers.CDense(units=4, lreg=lreg)
        self.creluaf2 = Layers.CReLUAF()

        self.cdense3 = Layers.CDense(units=4, lreg=lreg)
        self.creluaf3 = Layers.CReLUAF()

        self.cdense4 = Layers.CDense(units=1, lreg=lreg)
        self.csplitreim = Layers.CSplitReIm()

    def call(self, inputs):
        """Call method."""

        x = self.cfugacitycov(inputs)
        x = self.cdense1(x)
        x = self.creluaf1(x)
        x = self.cdense2(x)
        x = self.creluaf2(x)
        x = self.cdense3(x)
        x = self.creluaf3(x)
        x = self.cdense4(x)
        x = self.csplitreim(x)

        return x

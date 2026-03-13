#!/usr/bin/env python

import sys

import tensorflow as tf

from misc.Decorators import *
from network.BaseLayers import *


class MultiScaleResNet(BaseLayers):
    def __init__(
        self,
        InputPH=None,
        Padding=None,
        NumOut=None,
        InitNeurons=None,
        ExpansionFactor=None,
        NumSubBlocks=None,
        NumBlocks=None,
        Suffix=None,
        UncType=None,
    ):
        super(MultiScaleResNet, self).__init__()
        if InputPH is None:
            print("ERROR: Input PlaceHolder cannot be empty!")
            sys.exit(0)
        self.InputPH = InputPH
        if InitNeurons is None:
            InitNeurons = 37
        if ExpansionFactor is None:
            ExpansionFactor = 2.0
        if NumSubBlocks is None:
            NumSubBlocks = 2
        if NumBlocks is None:
            NumBlocks = 1
        self.InitNeurons = InitNeurons
        self.ExpansionFactor = ExpansionFactor
        self.DropOutRate = 0.7
        self.NumSubBlocks = NumSubBlocks
        self.NumBlocks = NumBlocks
        if Suffix is None:
            Suffix = ""
        self.Suffix = Suffix
        if NumOut is None:
            NumOut = 1
        self.NumOut = NumOut
        self.currBlock = 0
        self.UncType = UncType
        if (
            self.UncType == "Aleatoric"
            or self.UncType == "Inlier"
            or self.UncType == "LinearSoftplus"
        ):
            self.NumOut *= 2

        self.kernel_size = (3, 3)
        self.strides = (2, 2)
        if Padding is None:
            Padding = "same"
        self.padding = Padding

    def ResizeConv(self, inputs, filters, kernel_size=None, activation=None, name=None):
        if kernel_size is None:
            kernel_size = self.kernel_size

        input_shape = inputs.get_shape().as_list()
        new_height = input_shape[1] * 2
        new_width = input_shape[2] * 2

        upsampled = tf.compat.v1.image.resize_bilinear(
            inputs,
            size=[new_height, new_width],
            align_corners=False,
            half_pixel_centers=False,
            name=(name + "_resize") if name else None,
        )

        output = self.Conv(
            inputs=upsampled,
            filters=filters,
            kernel_size=kernel_size,
            strides=(1, 1),
            activation=activation,
            name=(name + "_conv") if name else None,
        )
        return output

    def ResizeConvBNReLUBlock(self, inputs, filters, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.kernel_size

        net = self.ResizeConv(inputs, filters, kernel_size)
        net = self.BN(net)
        net = self.ReLU(net)
        return net

    @CountAndScope
    def ResBlock(self, inputs=None, filters=None, kernel_size=None, strides=None, padding=None):
        if kernel_size is None:
            kernel_size = self.kernel_size
        if strides is None:
            strides = self.strides
        if padding is None:
            padding = self.padding
        net = self.ConvBNReLUBlock(
            inputs=inputs, filters=filters, padding=padding, strides=(1, 1)
        )
        net = self.Conv(
            inputs=net, filters=filters, padding=padding, strides=(1, 1), activation=None
        )
        net = self.BN(inputs=net)
        net = tf.add(net, inputs)
        net = self.ReLU(inputs=net)
        return net

    @CountAndScope
    def GlobalBroadcastGate(self, context_inputs=None, target_inputs=None, target_filters=None, name="global_gate"):
        context = tf.reduce_mean(
            context_inputs,
            axis=[1, 2],
            keepdims=True,
            name=name + "_mean",
        )
        gate = self.Conv(
            inputs=context,
            filters=target_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation=None,
            name=name + "_proj",
        )
        gate = tf.math.sigmoid(gate, name=name + "_sigmoid")
        return tf.multiply(target_inputs, gate, name=name + "_scale")

    @CountAndScope
    def ResNetBlock(self, inputs):
        num_filters = self.InitNeurons
        net = self.ConvBNReLUBlock(inputs=inputs, filters=num_filters, kernel_size=(7, 7))

        num_filters = int(num_filters * self.ExpansionFactor)
        net = self.ConvBNReLUBlock(inputs=net, filters=num_filters, kernel_size=(5, 5))

        for _ in range(self.NumSubBlocks):
            net = self.ResBlock(inputs=net, filters=num_filters)
            num_filters = int(num_filters * self.ExpansionFactor)
            net = self.Conv(inputs=net, filters=num_filters)

        bottleneck_context = net

        nets = []
        for count in range(self.NumSubBlocks):
            net = self.ResBlock(inputs=net, filters=num_filters)
            num_filters = int(num_filters / self.ExpansionFactor)
            net = self.ResizeConv(inputs=net, filters=num_filters)
            if count == 1:
                net = self.GlobalBroadcastGate(
                    context_inputs=bottleneck_context,
                    target_inputs=net,
                    target_filters=num_filters,
                    name="global_gate_4x",
                )

        net_out = self.Conv(
            inputs=net,
            filters=self.NumOut,
            kernel_size=(7, 7),
            strides=(1, 1),
            activation=None,
        )
        nets.append(net_out)
        print(f"[*] Decoder Out 1: {net_out.shape}")

        num_filters = int(num_filters / self.ExpansionFactor)
        net = self.ResizeConvBNReLUBlock(inputs=net, filters=num_filters, kernel_size=(5, 5))

        net_out = self.Conv(
            inputs=net,
            filters=self.NumOut,
            kernel_size=(7, 7),
            strides=(1, 1),
            activation=None,
        )
        nets.append(net_out)
        print(f"[*] Decoder Out 2: {net_out.shape}")

        num_filters = int(num_filters / self.ExpansionFactor)
        net = self.ResizeConvBNReLUBlock(inputs=net, filters=num_filters, kernel_size=(7, 7))
        print(f"[*] Upsample Final 2 output shape: {net.shape}")

        net = self.Conv(
            inputs=net,
            filters=self.NumOut,
            kernel_size=(7, 7),
            strides=(1, 1),
            activation=None,
        )
        nets.append(net)
        print(f"[*] Main Output shape: {net.shape}")

        return nets

    def Network(self):
        out_now = self.InputPH
        for count in range(self.NumBlocks):
            with tf.compat.v1.variable_scope("EncoderDecoderBlock" + str(count) + self.Suffix):
                out_now = self.ResNetBlock(out_now)
                self.currBlock += 1
        return out_now

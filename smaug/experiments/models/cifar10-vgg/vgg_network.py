#!/usr/bin/env python

"""Example for creating the CIFAR10-VGG network."""

import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_vgg_model():
  with sg.Graph(name="vgg_smv", backend="SMV") as graph:
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1, 32, 32, 3)))
    conv0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((64, 3, 3, 3)))
    conv1_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((128, 3, 3, 64)))
    conv2_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((128, 3, 3, 128)))
    conv3_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((128, 3, 3, 128)))
    conv4_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((256, 3, 3, 128)))
    conv5_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((256, 3, 3, 256)))
    conv6_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((256, 3, 3, 256)))
    conv7_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((512, 3, 3, 256)))
    conv8_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((512, 3, 3, 512)))
    conv9_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((512, 3, 3, 512)))
    fc0_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((512, 2048)))
    fc1_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((10, 512)))

    act = sg.input_data(input_tensor)
    act = sg.nn.convolution(
        act, conv0_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.convolution(
        act, conv1_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = sg.nn.convolution(
        act, conv2_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.convolution(
        act, conv3_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = sg.nn.convolution(
        act, conv4_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.convolution(
        act, conv5_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.convolution(
        act, conv6_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = sg.nn.convolution(
        act, conv7_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.convolution(
        act, conv8_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.convolution(
        act, conv9_tensor, stride=[1, 1], padding="same", activation="relu")
    act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = sg.nn.mat_mul(act, fc0_tensor, activation="relu")
    act = sg.nn.mat_mul(act, fc1_tensor)
    return graph

if __name__ != "main":
  graph = create_vgg_model()
  graph.print_summary()
  graph.write_graph()

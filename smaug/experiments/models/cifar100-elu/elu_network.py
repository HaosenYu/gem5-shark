#!/usr/bin/env python

"""Example for creating the CIFAR100-ELU network."""

import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float32)

def create_elu_model():
  with sg.Graph(name="elu_ref", backend="Reference",
                mem_policy=sg.AllDma) as graph:
    # sg.Tensors and kernels are initialized as NCHW layout.
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1, 32, 32, 3)))
    conv0_stack0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((192, 5, 5, 3)))
    conv0_stack1_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((192, 1, 1, 192)))
    conv1_stack1_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((240, 3, 3, 192)))
    conv0_stack2_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((240, 1, 1, 240)))
    conv1_stack2_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((260, 2, 2, 240)))
    conv0_stack3_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((260, 1, 1, 260)))
    conv1_stack3_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((280, 2, 2, 260)))
    conv0_stack4_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((280, 1, 1, 280)))
    conv1_stack4_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((300, 2, 2, 280)))
    conv0_stack5_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((300, 1, 1, 300)))
    conv0_stack6_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((100, 1, 1, 300)))

    act = sg.input_data(input_tensor, name="input")
    act = sg.nn.convolution(
        act, conv0_stack0_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv0_stack0")
    act = sg.nn.max_pool(
        act, pool_size=[2, 2], stride=[2, 2], name="pool_stack0")
    act = sg.nn.convolution(
        act, conv0_stack1_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv0_stack1")
    act = sg.nn.convolution(
        act, conv1_stack1_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv1_stack1")
    act = sg.nn.max_pool(
        act, pool_size=[2, 2], stride=[2, 2], name="pool_stack1")
    act = sg.nn.convolution(
        act, conv0_stack2_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv0_stack2")
    act = sg.nn.convolution(
        act, conv1_stack2_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv1_stack2")
    act = sg.nn.max_pool(
        act, pool_size=[2, 2], stride=[2, 2], name="pool_stack2")
    act = sg.nn.convolution(
        act, conv0_stack3_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv0_stack3")
    act = sg.nn.convolution(
        act, conv1_stack3_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv1_stack3")
    act = sg.nn.max_pool(
        act, pool_size=[2, 2], stride=[2, 2], name="pool_stack3")
    act = sg.nn.convolution(
        act, conv0_stack4_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv0_stack4")
    act = sg.nn.convolution(
        act, conv1_stack4_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv0_stack4")
    act = sg.nn.max_pool(
        act, pool_size=[2, 2], stride=[2, 2], name="pool_stack4")
    act = sg.nn.convolution(
        act, conv0_stack5_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv0_stack5")
    act = sg.nn.convolution(
        act, conv0_stack6_tensor, stride=[1, 1], padding="same",
        activation="elu", name="conv0_stack6")
    return graph

if __name__ != "main":
  graph = create_elu_model()
  graph.print_summary()
  graph.write_graph()

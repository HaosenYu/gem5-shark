#!/usr/bin/env python

"""Example for creating the ResNet-50 network."""

import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.008).astype(np.float16)

def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names

  Returns:
    Output tensor for the block.
  """
  filters0, filters1, filters2 = filters
  conv_name_base = 'res' + str(stage) + block
  bn_name_base = 'bn' + str(stage) + block
  add_name = 'add' + str(stage) + "_" + block
  relu_name = 'relu' + str(stage) + "_" + block

  # Tensors
  input_tensor_chans = input_tensor.dims(
      3) if input_tensor.shape.layout == sg.NHWC else input_tensor.dims(1)
  conv0_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (filters0, 1, 1, input_tensor_chans)))
  bn0_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters0)))
  bn0_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters0)))
  bn0_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters0)))
  bn0_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters0)))
  conv1_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (filters1, kernel_size, kernel_size, filters0)))
  bn1_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters1)))
  bn1_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters1)))
  bn1_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters1)))
  bn1_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters1)))
  conv2_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (filters2, 1, 1, filters1)))
  bn2_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn2_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn2_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn2_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))

  x = sg.nn.convolution(
      input_tensor, conv0_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2a')
  x = sg.nn.batch_norm(
      x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
      activation="relu", name=bn_name_base + '_2a')
  x = sg.nn.convolution(
      x, conv1_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2b')
  x = sg.nn.batch_norm(
      x, bn1_mean_tensor, bn1_var_tensor, bn1_gamma_tensor, bn1_beta_tensor,
      activation="relu", name=bn_name_base + '_2b')
  x = sg.nn.convolution(
      x, conv2_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2c')
  x = sg.nn.batch_norm(
      x, bn2_mean_tensor, bn2_var_tensor, bn2_gamma_tensor, bn2_beta_tensor,
      name=bn_name_base + '_2c')
  x = sg.math.add(x, input_tensor, name=add_name)
  x = sg.nn.relu(x, name=relu_name)
  return x

def conv_block(
    input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
  """A block that has a conv layer at shortcut.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    use_l2_regularizer: whether to use L2 regularizer on Conv layer.

  Returns:
    Output tensor for the block.
  """
  filters0, filters1, filters2 = filters
  conv_name_base = 'res' + str(stage) + block
  bn_name_base = 'bn' + str(stage) + block
  add_name = 'add' + str(stage) + "_" + block
  relu_name = 'relu' + str(stage) + "_" + block

  # sg.Tensors
  input_tensor_chans = input_tensor.dims(
      3) if input_tensor.shape.layout == sg.NHWC else input_tensor.dims(1)
  conv0_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (filters0, 1, 1, input_tensor_chans)))
  bn0_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters0)))
  bn0_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters0)))
  bn0_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters0)))
  bn0_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters0)))
  conv1_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (filters1, kernel_size, kernel_size, filters0)))
  bn1_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters1)))
  bn1_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters1)))
  bn1_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters1)))
  bn1_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters1)))
  conv2_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (filters2, 1, 1, filters1)))
  bn2_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn2_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn2_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn2_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  conv3_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (filters2, 1, 1, input_tensor_chans)))
  bn3_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn3_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn3_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))
  bn3_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, filters2)))

  x = sg.nn.convolution(
      input_tensor, conv0_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2a')
  x = sg.nn.batch_norm(
      x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
      activation="relu")
  x = sg.nn.convolution(
      x, conv1_tensor, stride=strides, padding="same",
      name=conv_name_base + '_2b')
  x = sg.nn.batch_norm(
      x, bn1_mean_tensor, bn1_var_tensor, bn1_gamma_tensor, bn1_beta_tensor,
      activation="relu", name=bn_name_base + '_2b')
  x = sg.nn.convolution(
      x, conv2_tensor, stride=[1, 1], padding="same",
      name=conv_name_base + '_2c')
  x = sg.nn.batch_norm(
      x, bn2_mean_tensor, bn2_var_tensor, bn2_gamma_tensor, bn2_beta_tensor,
      name=bn_name_base + '_2c')
  shortcut = sg.nn.convolution(
      input_tensor, conv3_tensor, stride=strides, padding="same",
      name=conv_name_base + '_1')
  shortcut = sg.nn.batch_norm(
      shortcut, bn3_mean_tensor, bn3_var_tensor, bn3_gamma_tensor,
      bn3_beta_tensor, name=bn_name_base + '_1')
  x = sg.math.add(x, shortcut, name=add_name)
  x = sg.nn.relu(x, name=relu_name)
  return x

def create_resnet50():
  with sg.Graph(name="resnet_smv", backend="SMV") as graph:
    # sg.Tensors and kernels are initialized as sg.NCHW layout.
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1, 225, 225, 3)))
    conv0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((64, 7, 7, 3)))
    bn0_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 64)))
    bn0_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 64)))
    bn0_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 64)))
    bn0_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 64)))
    fc_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((10, 7 * 7 * 2048)))

    x = sg.input_data(input_tensor, name="input")
    x = sg.nn.convolution(
        x, conv0_tensor, stride=[2, 2], padding="same", name="conv0")
    x = sg.nn.batch_norm(
        x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
        activation="relu", name="bn0")
    x = sg.nn.max_pool(x, pool_size=[3, 3], stride=[2, 2], name="pool")

    # Four resnet blocks.
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = sg.nn.mat_mul(x, fc_tensor, name="fc")
    return graph

if __name__ != "main":
  resnet50 = create_resnet50()
  resnet50.print_summary()
  resnet50.write_graph()

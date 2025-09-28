#!/usr/bin/env python

"""Create the LeNet-5 network."""
import numpy as np
import smaug as sg
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0
    x_test = x_test[..., np.newaxis] / 255.0
    return x_train.astype(np.float16), y_train, x_test.astype(np.float16), y_test

'''
def load_specific_tensor_data(shape):
    real_data = load_mnist_data()
    if len(shape) == 4:
        return real_data[:shape[0], :shape[1], :shape[2], :shape[3]]
    else:
        raise ValueError("Unsupported shape for real data.")
'''

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_lenet5_model_with_mnist():
  with sg.Graph(name="lenet5_smv", backend="SMV") as graph:
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=x_train)
    conv0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 1)))
    conv1_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 32)))
    fc0_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 4608)))
    fc1_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((10, 128)))

    act = sg.input_data(input_tensor)
    act = sg.nn.convolution(
        act, conv0_tensor, stride=[1, 1], padding="valid", activation="relu")
    act = sg.nn.convolution(
        act, conv1_tensor, stride=[1, 1], padding="valid", activation="relu")
    act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
    act = sg.nn.mat_mul(act, fc0_tensor, activation="relu")
    act = sg.nn.mat_mul(act, fc1_tensor)
    # return graph
    return graph, (x_train, y_train, x_test, y_test)

if __name__ != "main":
  # graph = create_lenet5_model_with_mnist()
  graph, mnist_data = create_lenet5_model_with_mnist()
  x_train, y_train, x_test, y_test = mnist_data
  
  graph.print_summary()
  print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
  print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
  
  graph.write_graph()

#!/usr/bin/env python

import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float32)

def create_lstm_model():
  with sg.Graph(name="lstm_ref", backend="Reference") as graph:
    input_tensor = sg.Tensor(
        data_layout=sg.NTC, tensor_data=generate_random_data((1, 4, 32)))
    # sg.Tensors and kernels are initialized as NC layout.
    # Layer 1 of LSTM.
    w0 = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 32)))
    u0 = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 32)))
    # Layer 2 of LSTM.
    w1 = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 32)))
    u1 = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 32)))

    # Inputs specified in shape (batch, timestep, size)
    inputs = sg.input_data(input_tensor, name="input")
    lstm_layer0 = sg.nn.LSTM([w0, u0], name="lstm0")
    lstm_layer1 = sg.nn.LSTM([w1, u1], name="lstm1")
    outputs, state = lstm_layer0(inputs)
    outputs, state = lstm_layer1(outputs)
    return graph

if __name__ != "main":
  graph = create_lstm_model()
  graph.print_summary()
  graph.write_graph()

#!/usr/bin/env python

import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_lstm_model():
  with sg.Graph(name="bidi_lstm_smv", backend="SMV") as graph:
    input_tensor = sg.Tensor(
        data_layout=sg.NTC, tensor_data=generate_random_data((1, 4, 32)))
    # sg.Tensors and kernels are initialized as NC layout.
    # Weights of forward LSTM.
    w_f = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 32)))
    u_f = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 32)))
    # Weights of backward LSTM.
    w_b = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 32)))
    u_b = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((128, 32)))

    # Inputs specified in shape (batch, timestep, size)
    inputs = sg.input_data(input_tensor, name="input")
    bidi_lstm = sg.nn.BidirectionalLSTM([w_f, u_f], [w_b, u_b],
                                        name="bidi_lstm")
    outputs, state_fwd, state_bwd = bidi_lstm(inputs)
    return graph

if __name__ != "main":
  graph = create_lstm_model()
  graph.print_summary()
  graph.write_graph()

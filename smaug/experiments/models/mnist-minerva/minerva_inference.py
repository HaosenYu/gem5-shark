#!/usr/bin/env python

import numpy as np
import smaug as sg
from tensorflow.keras.datasets import mnist

def load_mnist_images_keras(num_samples=8):
    """
    使用 tensorflow.keras.datasets.mnist 直接加载测试集图像。
    返回 (X, Y):
        - X.shape = (num_samples,28,28,1), dtype=float16
        - Y.shape = (num_samples,)
    """
    
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 255.0  # 与训练一致
    # 取前 num_samples 张
    x_test = x_test[:num_samples]  # shape=(N,28,28)
    y_test = y_test[:num_samples]  # shape=(N,)

    # 加最后一个通道维度 -> (N,28,28,1)
    x_test = np.expand_dims(x_test, axis=-1)  # (N,28,28,1)
    # 转为 float16
    x_test = x_test.astype(np.float16)

    return x_test, y_test

def create_minerva_mnist_model(batch_size=8):
    # 加载若干张测试图片 (N,28,28,1)
    input_data, labels = load_mnist_images_keras(num_samples=batch_size)
    print(f"Loaded {batch_size} test image(s). Shape={input_data.shape}, dtype={input_data.dtype}")
    print("Labels =", labels)

    # 加载 .npz 权重，暂时忽略bias
    params = np.load("mnist_weights.npz")
    fc0_w = params["fc0_w"].astype(np.float16)  # (256,784)
    fc0_b = params["fc0_b"].astype(np.float16)  # (256,)
    fc1_w = params["fc1_w"].astype(np.float16)  # (256,256)
    fc1_b = params["fc1_b"].astype(np.float16)  # (256,)
    fc2_w = params["fc2_w"].astype(np.float16)  # (256,256)
    fc2_b = params["fc2_b"].astype(np.float16)  # (256,)
    fc3_w = params["fc3_w"].astype(np.float16)  # (10,256)
    fc3_b = params["fc3_b"].astype(np.float16)  # (10,)

    # 在 Smaug 中构建网络
    with sg.Graph(name="minerva_smv", backend="SMV") as graph:
        # a) 输入 Tensor
        input_tensor = sg.Tensor(
            data_layout=sg.NHWC,  # (N,H,W,C)
            tensor_data=input_data
        )

        # b) 各层权重、偏置 Tensor
        fc0_w_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc0_w)
        fc1_w_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc1_w)
        fc2_w_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc2_w)
        fc3_w_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc3_w)

        # bias 
        fc0_b_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc0_b.reshape(1,-1))
        fc1_b_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc1_b.reshape(1,-1))
        fc2_b_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc2_b.reshape(1,-1))
        fc3_b_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc3_b.reshape(1,-1))

        # c) 依次搭建 4 层全连接
        act = sg.input_data(input_tensor)
        # FC0: (N,784)->(N,256)
        act = sg.nn.mat_mul(act, fc0_w_tensor, activation="relu")

        # FC1: (N,256)->(N,256)
        act = sg.nn.mat_mul(act, fc1_w_tensor, activation="relu")

        # FC2: (N,256)->(N,256)
        act = sg.nn.mat_mul(act, fc2_w_tensor, activation="relu")

        # FC3: (N,256)->(N,10)
        act = sg.nn.mat_mul(act, fc3_w_tensor)  # 没有 activation
        
        '''
        No Bias
        act = sg.input_data(input_tensor)
        # FC0: (N,784)->(N,256)
        act = sg.nn.mat_mul(act, fc0_w_tensor)
        act = sg.nn.add_bias(act, fc0_b_tensor, activation="relu")

        # FC1
        act = sg.nn.mat_mul(act, fc1_w_tensor)
        act = sg.nn.add_bias(act, fc1_b_tensor, activation="relu")

        # FC2
        act = sg.nn.mat_mul(act, fc2_w_tensor)
        act = sg.nn.add_bias(act, fc2_b_tensor, activation="relu")

        # FC3: (N,256)->(N,10)
        act = sg.nn.mat_mul(act, fc3_w_tensor)
        act = sg.nn.add_bias(act, fc3_b_tensor)
        '''

        return graph, act, labels  # 返回最终输出张量和真实标签，用于后处理

if __name__ == "__main__":
    # 创建 SMAUG 计算图
    graph, output_tensor, true_labels = create_minerva_mnist_model(batch_size=1)
    graph.print_summary()
    graph.write_graph()
    


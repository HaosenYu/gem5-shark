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
    x_test = x_test.astype("float32") / 255.0  # 归一化
    x_test = x_test[:num_samples]  # 选取部分样本
    y_test = y_test[:num_samples]

    # 调整大小 (28x28 -> 32x32)
    x_test_resized = np.zeros((num_samples, 32, 32, 1), dtype=np.float16)
    x_test_resized[:, 2:30, 2:30, :] = np.expand_dims(x_test, axis=-1)  # 居中填充

    return x_test_resized, y_test

def create_cnn10_mnist_model(batch_size=8):
    # 加载测试数据
    input_data, labels = load_mnist_images_keras(num_samples=batch_size)
    print(f"Loaded {batch_size} test image(s). Shape={input_data.shape}, dtype={input_data.dtype}")
    print("Labels =", labels)

    # 加载 CNN10 训练得到的 .npz 权重
    params = np.load("cnn10_weights.npz")

    # 读取卷积层权重（NHWC 格式）
    conv0_w = params["conv0_w"].astype(np.float16)  # (32, 3, 3, 1)
    conv1_w = params["conv1_w"].astype(np.float16)  # (32, 3, 3, 32)
    conv2_w = params["conv2_w"].astype(np.float16)  # (64, 3, 3, 32)
    conv3_w = params["conv3_w"].astype(np.float16)  # (64, 3, 3, 64)

    # 读取全连接层权重
    fc0_w = params["fc0_w"].astype(np.float16)  # (512, 4096)
    fc1_w = params["fc1_w"].astype(np.float16)  # (10, 512)

    # 在 Smaug 中构建 CNN10 网络
    with sg.Graph(name="cnn10_smv", backend="SMV") as graph:
        # 1. 输入张量
        input_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=input_data)

        # 2. 卷积层 + BatchNorm + ReLU
        conv0_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=conv0_w)
        act = sg.nn.convolution(input_tensor, conv0_tensor, stride=[1, 1], padding="same", activation="relu")

        conv1_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=conv1_w)
        act = sg.nn.convolution(act, conv1_tensor, stride=[1, 1], padding="same", activation="relu")
        act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])  # MaxPool (2x2)

        conv2_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=conv2_w)
        act = sg.nn.convolution(act, conv2_tensor, stride=[1, 1], padding="same", activation="relu")

        conv3_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=conv3_w)
        act = sg.nn.convolution(act, conv3_tensor, stride=[1, 1], padding="same", activation="relu")
        act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])  # MaxPool (2x2)

        # 3. 全连接层
        fc0_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc0_w)
        fc1_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=fc1_w)

        act = sg.nn.mat_mul(act, fc0_tensor, activation="relu")  # FC0
        act = sg.nn.mat_mul(act, fc1_tensor)  # FC1 (Logits)

        return graph, act, labels  # 返回计算图、输出张量、真实标签

if __name__ == "__main__":
    # 创建 SMAUG 计算图
    graph, output_tensor, true_labels = create_cnn10_mnist_model(batch_size=1)
    graph.print_summary()
    graph.write_graph()


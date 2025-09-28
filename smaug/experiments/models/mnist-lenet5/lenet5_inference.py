#!/usr/bin/env python

import numpy as np
import smaug as sg
from tensorflow.keras.datasets import mnist



def load_mnist_images_keras(num_samples=1):
    """
    使用 tensorflow.keras.datasets.mnist 随机加载测试集图像。
    返回 (X, Y):
        - X.shape = (num_samples,28,28,1), dtype=float16
        - Y.shape = (num_samples,)
    """
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float16") / 255.0  # 与训练一致

    # 随机采样 num_samples 个索引
    num_total = x_test.shape[0]
    random_indices = np.random.choice(num_total, num_samples, replace=False)

    # 根据随机索引提取样本
    x_test = x_test[random_indices]  # shape=(num_samples,28,28)
    y_test = y_test[random_indices]  # shape=(num_samples,)

    # 加最后一个通道维度 -> (num_samples,28,28,1)
    x_test = np.expand_dims(x_test, axis=-1)  # shape=(num_samples,28,28,1)
    # 转为 float16
    x_test = x_test.astype(np.float16)

    return x_test, y_test
    
def create_lenet5_mnist_model(batch_size=1):
    # 加载若干张测试图片 (N,28,28,1)
    input_data, labels = load_mnist_images_keras(num_samples=batch_size)
    print(f"Loaded {batch_size} test image(s). Shape={input_data.shape}, dtype={input_data.dtype}")
    print("Labels =", labels)
    
    # 加载预训练权重
    weights = np.load("lenet5_weights.npz")
    
    # 在 Smaug 中构建网络
    with sg.Graph(name="lenet5_smv", backend="SMV") as graph:
        # 进行转置
        # conv0_w = np.transpose(weights["conv0_w"], (1, 2, 3, 0))  # => (3,3,1,16)
        # conv1_w = np.transpose(weights["conv1_w"], (1, 2, 3, 0))  # => (3,3,16,16)

        conv0_tensor = sg.Tensor(
            data_layout=sg.NHWC,
            tensor_data=weights["conv0_w"].astype(np.float16)
        )
        conv1_tensor = sg.Tensor(
            data_layout=sg.NHWC,
            tensor_data=weights["conv1_w"].astype(np.float16)
        )
        fc0_tensor = sg.Tensor(
            data_layout=sg.NC,
            tensor_data=weights["fc0_w"].astype(np.float16) # => (128,4608)
        )
        fc1_tensor = sg.Tensor(
            data_layout=sg.NC,
            tensor_data=weights["fc1_w"].astype(np.float16) # => (10,128)
        )
        
        print("conv0_w (original):", weights["conv0_w"].shape)
        print("conv0_w (transposed):", conv0_tensor.shape)
        print("conv1_w (transposed):", conv1_tensor.shape)
        # print("fc0_w :", weights["fc0_w"].shape)
        # print("fc1_w :", weights["fc1_w"].shape)
        # assert input_tensor.shape[-1] == conv0_tensor.shape[2], "Input channel mismatch!"
        
        # input_data = np.zeros((1, 28, 28, 1), dtype=np.float16)
        input_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=input_data,name="input_tensor")
        act = sg.input_data(input_tensor)
        act = sg.nn.convolution(act, conv0_tensor, stride=[1, 1], padding="valid", activation="relu")
        act = sg.nn.convolution(act, conv1_tensor, stride=[1, 1], padding="valid", activation="relu")
        act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
        act = sg.nn.mat_mul(act, fc0_tensor, activation="relu")
        act = sg.nn.mat_mul(act, fc1_tensor)
        
        act.name = "output_tensor"
        graph.input_name = input_tensor.name
        graph.output_name = act.name

    return graph

# 求推理结果的伪代码逻辑，暂时注释
def inference_lenet5(graph, input_image):
    # 对单张 (1, 28, 28, 1) 的图片做一次推理, 返回 shape=[1, 10] 的输出向量（logits）。

    # 获取输入和输出名称
    input_name = graph.input_name
    output_name = graph.output_name

    # 设置输入张量
    print(f"Setting input tensor: {input_name}, Shape: {input_image.shape}")
    graph.set_tensor(input_name, input_image.astype(np.float16))  # 设置输入数据
    
    graph.compile()
    graph.run()

    # 获取输出数据
    print(f"Getting output tensor: {output_name}")
    output_data = graph.get_tensor_data(output_name)  # 获取输出张量数据
    return output_data


if __name__ == "__main__":
    # 创建 SMAUG 计算图
    graph = create_lenet5_mnist_model(batch_size=1)
    graph.print_summary()
    graph.write_graph()
    
    '''
    # 使用 tensorflow.keras.datasets 加载 MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 将测试集图像缩放到 [0,1]
    x_test = x_test.astype(np.float16) / 255.0
    
    # 逐张图片推理并计算准确率
    correct = 0
    total = x_test.shape[0]  # 10000
    
    for i in range(total):
        # 取第 i 张图片, shape=(28,28)
        img = x_test[i]
        # 转成 (1, 28, 28, 1) 方便在 NHWC 下推理
        img = np.expand_dims(img, axis=(0, 3))
        # print("input_image shape:", img.shape)
        # 推理
        logits = inference_lenet5(graph, img)  # shape=(1,10)
        
        # argmax 得到预测类别
        pred_label = np.argmax(logits[0])
        
        # 对比真实标签
        if pred_label == y_test[i]:
            correct += 1
        
        # 实时查看中间预测
        # if i % 1000 == 0:
        #     print(f"Sample {i}, pred={pred_label}, gt={y_test[i]}")
    
    accuracy = correct / total
    print(f"Test Accuracy on MNIST: {accuracy:.4f}")
    '''

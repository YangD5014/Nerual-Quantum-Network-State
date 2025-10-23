"""
神经网络基础知识的Python代码示例
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_classification, make_regression, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重复
np.random.seed(42)
torch.manual_seed(42)

# ==============================
# 1. 基本神经网络结构
# ==============================

class SimpleNeuralNetwork(nn.Module):
    """
    简单的前馈神经网络
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        """
        初始化神经网络
        
        参数:
            input_size: 输入层大小
            hidden_sizes: 隐藏层大小列表
            output_size: 输出层大小
            activation: 激活函数 ('relu', 'sigmoid', 'tanh')
        """
        super(SimpleNeuralNetwork, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 添加激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

def visualize_neural_network(model, input_size, hidden_sizes, output_size):
    """
    可视化神经网络结构
    
    参数:
        model: 神经网络模型
        input_size: 输入层大小
        hidden_sizes: 隐藏层大小列表
        output_size: 输出层大小
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 层的位置
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    layer_positions = np.arange(len(layer_sizes))
    
    # 绘制神经元和连接
    for i, size in enumerate(layer_sizes):
        # 计算神经元位置
        y_positions = np.linspace(-size/2, size/2, size)
        
        # 绘制神经元
        for y in y_positions:
            circle = plt.Circle((i, y), 0.1, color='skyblue', ec='black', zorder=4)
            ax.add_patch(circle)
        
        # 绘制连接
        if i < len(layer_sizes) - 1:
            next_size = layer_sizes[i + 1]
            next_y_positions = np.linspace(-next_size/2, next_size/2, next_size)
            
            for y in y_positions:
                for next_y in next_y_positions:
                    ax.plot([i, i + 1], [y, next_y], 'gray', alpha=0.3, zorder=1)
    
    # 设置图形属性
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    ax.set_ylim(-max(layer_sizes)/2 - 0.5, max(layer_sizes)/2 + 0.5)
    ax.set_aspect('equal')
    ax.set_title('神经网络结构可视化')
    ax.set_xticks(layer_positions)
    ax.set_xticklabels(['输入层'] + [f'隐藏层 {i+1}' for i in range(len(hidden_sizes))] + ['输出层'])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def example_neural_network_structure():
    """
    神经网络结构示例
    """
    print("=== 神经网络结构示例 ===")
    
    # 创建神经网络
    input_size = 10
    hidden_sizes = [20, 15]
    output_size = 1
    model = SimpleNeuralNetwork(input_size, hidden_sizes, output_size, activation='relu')
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    # 可视化神经网络结构
    visualize_neural_network(model, input_size, hidden_sizes, output_size)
    
    # 测试前向传播
    x = torch.randn(5, input_size)  # 5个样本，每个样本10个特征
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出值: {output.detach().numpy().flatten()}")

# ==============================
# 2. 激活函数
# ==============================

def visualize_activation_functions():
    """
    可视化不同的激活函数
    """
    print("\n=== 激活函数可视化 ===")
    
    # 创建输入值
    x = torch.linspace(-5, 5, 100)
    
    # 计算不同激活函数的输出
    sigmoid = torch.sigmoid(x)
    tanh = torch.tanh(x)
    relu = torch.relu(x)
    leaky_relu = F.leaky_relu(x, negative_slope=0.1)
    elu = F.elu(x)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 绘制Sigmoid函数
    axes[0, 0].plot(x.numpy(), sigmoid.numpy())
    axes[0, 0].set_title('Sigmoid')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('σ(x)')
    axes[0, 0].grid(True)
    
    # 绘制Tanh函数
    axes[0, 1].plot(x.numpy(), tanh.numpy())
    axes[0, 1].set_title('Tanh')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('tanh(x)')
    axes[0, 1].grid(True)
    
    # 绘制ReLU函数
    axes[0, 2].plot(x.numpy(), relu.numpy())
    axes[0, 2].set_title('ReLU')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('ReLU(x)')
    axes[0, 2].grid(True)
    
    # 绘制Leaky ReLU函数
    axes[1, 0].plot(x.numpy(), leaky_relu.numpy())
    axes[1, 0].set_title('Leaky ReLU')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('LeakyReLU(x)')
    axes[1, 0].grid(True)
    
    # 绘制ELU函数
    axes[1, 1].plot(x.numpy(), elu.numpy())
    axes[1, 1].set_title('ELU')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('ELU(x)')
    axes[1, 1].grid(True)
    
    # 隐藏最后一个子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==============================
# 3. 损失函数
# ==============================

def visualize_loss_functions():
    """
    可视化不同的损失函数
    """
    print("\n=== 损失函数可视化 ===")
    
    # 创建预测值和真实值
    y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])
    y_pred = torch.linspace(0, 1, 100)
    
    # 计算不同损失函数的值
    mse_loss = []
    bce_loss = []
    
    for pred in y_pred:
        # MSE损失 (对于二元分类)
        mse = F.mse_loss(torch.tensor([pred, pred, pred, pred]), y_true, reduction='mean')
        mse_loss.append(mse.item())
        
        # 二元交叉熵损失
        bce = F.binary_cross_entropy(torch.tensor([pred, pred, pred, pred]), y_true, reduction='mean')
        bce_loss.append(bce.item())
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制MSE损失
    ax1.plot(y_pred.numpy(), mse_loss)
    ax1.set_title('均方误差 (MSE)')
    ax1.set_xlabel('预测值')
    ax1.set_ylabel('损失')
    ax1.grid(True)
    
    # 绘制二元交叉熵损失
    ax2.plot(y_pred.numpy(), bce_loss)
    ax2.set_title('二元交叉熵损失')
    ax2.set_xlabel('预测值')
    ax2.set_ylabel('损失')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================
# 4. 优化算法
# ==============================

def compare_optimizers():
    """
    比较不同优化算法的性能
    """
    print("\n=== 优化算法比较 ===")
    
    # 创建简单的二次函数作为优化目标
    def quadratic_function(x):
        return x**2 + 10*torch.sin(x)
    
    # 创建输入值
    x = torch.linspace(-10, 10, 100)
    y = quadratic_function(x)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制函数曲线
    ax.plot(x.numpy(), y.numpy(), 'b-', label='目标函数')
    
    # 初始化优化器
    optimizers = {
        'SGD': optim.SGD,
        'SGD with Momentum': lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop
    }
    
    colors = ['r', 'g', 'm', 'c']
    
    # 测试不同优化器
    for i, (name, optimizer_class) in enumerate(optimizers.items()):
        # 初始化参数
        param = torch.tensor([8.0], requires_grad=True)
        optimizer = optimizer_class([param], lr=0.1)
        
        # 记录优化路径
        path_x = [param.item()]
        path_y = [quadratic_function(param).item()]
        
        # 优化过程
        for _ in range(50):
            optimizer.zero_grad()
            loss = quadratic_function(param)
            loss.backward()
            optimizer.step()
            
            path_x.append(param.item())
            path_y.append(quadratic_function(param).item())
        
        # 绘制优化路径
        ax.plot(path_x, path_y, 'o-', color=colors[i], label=name, markersize=4)
    
    # 设置图形属性
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('不同优化算法的性能比较')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================
# 5. 前馈神经网络应用
# ==============================

def train_feedforward_nn_classification():
    """
    使用前馈神经网络进行分类任务
    """
    print("\n=== 前馈神经网络分类任务 ===")
    
    # 生成分类数据集
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_redundant=5, 
        n_classes=2, 
        random_state=42
    )
    
    # 数据预处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    model = SimpleNeuralNetwork(
        input_size=20, 
        hidden_sizes=[64, 32], 
        output_size=1, 
        activation='relu'
    )
    
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 100
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_test))
        y_pred_class = (y_pred > 0.5).float()
        
        accuracy = accuracy_score(y_test, y_pred_class)
        print(f'测试集准确率: {accuracy:.4f}')
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred_class))
        
        # 绘制混淆矩阵
        cm = confusion_matrix(y_test, y_pred_class)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.show()
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True)
    plt.show()
    
    return model

def train_feedforward_nn_regression():
    """
    使用前馈神经网络进行回归任务
    """
    print("\n=== 前馈神经网络回归任务 ===")
    
    # 生成回归数据集
    X, y = make_regression(
        n_samples=1000, 
        n_features=10, 
        n_informative=8, 
        noise=0.1, 
        random_state=42
    )
    
    # 数据预处理
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    model = SimpleNeuralNetwork(
        input_size=10, 
        hidden_sizes=[64, 32], 
        output_size=1, 
        activation='relu'
    )
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 100
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'测试集均方误差: {mse:.4f}')
        
        # 绘制预测值与真实值的散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值与真实值对比')
        plt.grid(True)
        plt.show()
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True)
    plt.show()
    
    return model

# ==============================
# 6. 卷积神经网络
# ==============================

class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络
    """
    def __init__(self, input_channels=1, num_classes=10):
        """
        初始化CNN
        
        参数:
            input_channels: 输入通道数
            num_classes: 分类数量
        """
        super(SimpleCNN, self).__init__()
        
        # 卷积层1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, channels, height, width)
            
        返回:
            输出张量
        """
        # 卷积层1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 卷积层2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        return x

def visualize_cnn_feature_maps():
    """
    可视化CNN的特征图
    """
    print("\n=== CNN特征图可视化 ===")
    
    # 创建一个简单的图像（例如，一个圆形）
    image_size = 28
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(-1, 1, image_size)
    xx, yy = np.meshgrid(x, y)
    
    # 创建一个圆形图像
    image = np.zeros((image_size, image_size))
    mask = xx**2 + yy**2 <= 0.5**2
    image[mask] = 1.0
    
    # 转换为PyTorch张量
    image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    
    # 创建CNN模型
    model = SimpleCNN(input_channels=1, num_classes=10)
    
    # 获取特征图
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    
    # 前向传播
    output = model(image_tensor)
    
    # 可视化原始图像
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    # 可视化第一层卷积的特征图（只显示前8个）
    conv1_act = activation['conv1'].squeeze().numpy()
    for i in range(min(8, conv1_act.shape[0])):
        plt.subplot(2, 5, i+2)
        plt.imshow(conv1_act[i], cmap='viridis')
        plt.title(f'Conv1-{i+1}')
        plt.axis('off')
    
    # 可视化第二层卷积的特征图（只显示前8个）
    conv2_act = activation['conv2'].squeeze().numpy()
    for i in range(min(8, conv2_act.shape[0])):
        plt.subplot(2, 5, i+2)
        plt.imshow(conv2_act[i], cmap='viridis')
        plt.title(f'Conv2-{i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==============================
# 7. 循环神经网络
# ==============================

class SimpleRNN(nn.Module):
    """
    简单的循环神经网络
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, rnn_type='rnn'):
        """
        初始化RNN
        
        参数:
            input_size: 输入特征大小
            hidden_size: 隐藏层大小
            output_size: 输出大小
            num_layers: RNN层数
            rnn_type: RNN类型 ('rnn', 'lstm', 'gru')
        """
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 根据类型创建RNN层
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_length, input_size)
            
        返回:
            输出张量
        """
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播RNN
        if isinstance(self.rnn, nn.LSTM):
            # LSTM有隐藏状态和细胞状态
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            # RNN和GRU只有隐藏状态
            out, _ = self.rnn(x, h0)
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        
        return out

def example_rnn_sequence_prediction():
    """
    使用RNN进行序列预测示例
    """
    print("\n=== RNN序列预测示例 ===")
    
    # 生成正弦波数据
    seq_length = 20
    data_size = 1000
    
    # 创建正弦波
    x = np.linspace(0, 100, data_size)
    y = np.sin(x)
    
    # 创建序列数据
    sequences = []
    targets = []
    
    for i in range(data_size - seq_length):
        sequences.append(y[i:i+seq_length])
        targets.append(y[i+seq_length])
    
    # 转换为PyTorch张量
    sequences = torch.FloatTensor(sequences).unsqueeze(-1)  # 添加特征维度
    targets = torch.FloatTensor(targets).unsqueeze(-1)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    test_sequences = sequences[train_size:]
    test_targets = targets[train_size:]
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_sequences, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    model = SimpleRNN(input_size=1, hidden_size=32, output_size=1, num_layers=1, rnn_type='lstm')
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 50
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        # 使用测试集进行预测
        predictions = model(test_sequences)
        mse = mean_squared_error(test_targets, predictions)
        print(f'测试集均方误差: {mse:.4f}')
        
        # 绘制预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(test_targets.numpy(), label='真实值')
        plt.plot(predictions.numpy(), label='预测值')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.title('RNN序列预测结果')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True)
    plt.show()
    
    return model

# ==============================
# 8. 自注意力机制与Transformer
# ==============================

class SelfAttention(nn.Module):
    """
    自注意力机制
    """
    def __init__(self, embed_dim, num_heads=8):
        """
        初始化自注意力机制
        
        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
        """
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 查询、键、值的线性变换
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # 输出线性变换
        self.out_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_length, embed_dim)
            
        返回:
            输出张量
        """
        batch_size, seq_length, embed_dim = x.size()
        
        # 线性变换
        q = self.q_linear(x)  # (batch_size, seq_length, embed_dim)
        k = self.k_linear(x)  # (batch_size, seq_length, embed_dim)
        v = self.v_linear(x)  # (batch_size, seq_length, embed_dim)
        
        # 分割多头
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # (batch_size, num_heads, seq_length, seq_length)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重到值上
        context = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_length, head_dim)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)
        
        # 输出线性变换
        output = self.out_linear(context)  # (batch_size, seq_length, embed_dim)
        
        return output, attention_weights

def visualize_attention_weights():
    """
    可视化注意力权重
    """
    print("\n=== 注意力权重可视化 ===")
    
    # 创建一些随机序列数据
    batch_size = 1
    seq_length = 10
    embed_dim = 64
    
    x = torch.randn(batch_size, seq_length, embed_dim)
    
    # 创建自注意力机制
    attention = SelfAttention(embed_dim, num_heads=4)
    
    # 前向传播
    output, attention_weights = attention(x)
    
    # 可视化注意力权重
    attention_weights = attention_weights.squeeze(0).detach().numpy()  # (num_heads, seq_length, seq_length)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < attention_weights.shape[0]:
            # 绘制注意力权重矩阵
            im = ax.imshow(attention_weights[i], cmap='viridis')
            ax.set_title(f'注意力头 {i+1}')
            ax.set_xlabel('键位置')
            ax.set_ylabel('查询位置')
            
            # 添加颜色条
            plt.colorbar(im, ax=ax)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==============================
# 9. 图神经网络
# ==============================

class SimpleGNN(nn.Module):
    """
    简单的图神经网络
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化GNN
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super(SimpleGNN, self).__init__()
        
        # 图卷积层
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵，形状为 (num_nodes, input_dim)
            adj: 邻接矩阵，形状为 (num_nodes, num_nodes)
            
        返回:
            节点输出特征
        """
        # 第一层图卷积
        x = torch.matmul(adj, x)  # 邻接矩阵与节点特征相乘
        x = F.relu(self.conv1(x))
        
        # 第二层图卷积
        x = torch.matmul(adj, x)
        x = self.conv2(x)
        
        return x

def example_gnn_node_classification():
    """
    使用GNN进行节点分类示例
    """
    print("\n=== GNN节点分类示例 ===")
    
    # 创建一个简单的图（例如，一个环形图）
    num_nodes = 10
    adj = torch.zeros(num_nodes, num_nodes)
    
    # 创建环形连接
    for i in range(num_nodes):
        adj[i, (i+1)%num_nodes] = 1
        adj[(i+1)%num_nodes, i] = 1
    
    # 添加自环
    adj += torch.eye(num_nodes)
    
    # 归一化邻接矩阵
    degree = torch.sum(adj, dim=1)
    inv_degree = torch.pow(degree, -0.5)
    inv_degree[inv_degree == float('inf')] = 0
    adj = torch.diag(inv_degree) @ adj @ torch.diag(inv_degree)
    
    # 创建节点特征
    node_features = torch.randn(num_nodes, 16)
    
    # 创建节点标签（例如，奇数节点和偶数节点属于不同类别）
    node_labels = torch.tensor([i % 2 for i in range(num_nodes)])
    
    # 创建模型
    model = SimpleGNN(input_dim=16, hidden_dim=32, output_dim=2)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 100
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        
        # 前向传播
        outputs = model(node_features, adj)
        loss = criterion(outputs, node_labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        outputs = model(node_features, adj)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == node_labels).sum().item() / num_nodes
        print(f'节点分类准确率: {accuracy:.4f}')
    
    # 可视化图和分类结果
    plt.figure(figsize=(10, 8))
    
    # 创建节点位置（环形布局）
    pos = {}
    for i in range(num_nodes):
        angle = 2 * np.pi * i / num_nodes
        pos[i] = (np.cos(angle), np.sin(angle))
    
    # 绘制节点
    for i in range(num_nodes):
        color = 'red' if predicted[i] == 0 else 'blue'
        plt.scatter(pos[i][0], pos[i][1], s=200, color=color, zorder=2)
        plt.text(pos[i][0], pos[i][1], str(i), ha='center', va='center', color='white', fontweight='bold')
    
    # 绘制边
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adj[i, j] > 0:
                plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 'gray', alpha=0.5, zorder=1)
    
    plt.title('GNN节点分类结果 (红色: 类别0, 蓝色: 类别1)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True)
    plt.show()
    
    return model

# ==============================
# 10. 生成模型
# ==============================

class SimpleVAE(nn.Module):
    """
    简单的变分自编码器
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        初始化VAE
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            latent_dim: 潜在空间维度
        """
        super(SimpleVAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        编码
        
        参数:
            x: 输入数据
            
        返回:
            潜在变量的均值和方差
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        重新参数化技巧
        
        参数:
            mu: 均值
            log_var: 对数方差
            
        返回:
            采样的潜在变量
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        解码
        
        参数:
            z: 潜在变量
            
        返回:
            重构的数据
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据
            
        返回:
            重构的数据、均值、对数方差
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

def vae_loss_function(recon_x, x, mu, log_var):
    """
    VAE损失函数
    
    参数:
        recon_x: 重构的数据
        x: 原始数据
        mu: 均值
        log_var: 对数方差
        
    返回:
        损失值
    """
    # 重构损失
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD

def example_vae_mnist_generation():
    """
    使用VAE生成MNIST数字示例
    """
    print("\n=== VAE生成MNIST数字示例 ===")
    
    # 注意：这里我们使用简化的数据，而不是真实的MNIST数据集
    # 创建一些简化的"数字"数据
    num_samples = 1000
    input_dim = 64  # 8x8图像
    
    # 生成一些随机的"数字"图像
    data = torch.zeros(num_samples, input_dim)
    
    # 创建一些简单的模式
    for i in range(num_samples):
        digit_type = i % 10
        if digit_type == 0:  # 圆形
            center = 32
            for j in range(input_dim):
                row, col = j // 8, j % 8
                dist = np.sqrt((row - 4)**2 + (col - 4)**2)
                if dist < 2.5:
                    data[i, j] = 1.0
        elif digit_type == 1:  # 垂直线
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if col == 3:
                    data[i, j] = 1.0
        elif digit_type == 2:  # 水平线
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if row == 3:
                    data[i, j] = 1.0
        elif digit_type == 3:  # 对角线
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if row == col:
                    data[i, j] = 1.0
        elif digit_type == 4:  # 反对角线
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if row + col == 7:
                    data[i, j] = 1.0
        elif digit_type == 5:  # 十字
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if row == 3 or col == 3:
                    data[i, j] = 1.0
        elif digit_type == 6:  # 方形
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if (row == 2 or row == 5) and (col >= 2 and col <= 5):
                    data[i, j] = 1.0
                if (col == 2 or col == 5) and (row >= 2 and row <= 5):
                    data[i, j] = 1.0
        elif digit_type == 7:  # L形
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if row == 2 and col <= 5:
                    data[i, j] = 1.0
                if col == 5 and row >= 2:
                    data[i, j] = 1.0
        elif digit_type == 8:  # T形
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if row == 2 and col >= 2 and col <= 5:
                    data[i, j] = 1.0
                if col == 3 and row >= 2:
                    data[i, j] = 1.0
        elif digit_type == 9:  # 三角形
            for j in range(input_dim):
                row, col = j // 8, j % 8
                if row >= 2 and row <= 5 and col >= 2 and col <= 5:
                    if row + col >= 7 and row - col <= 1:
                        data[i, j] = 1.0
    
    # 添加一些噪声
    data += torch.rand_like(data) * 0.1
    data = torch.clamp(data, 0, 1)
    
    # 创建数据加载器
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    model = SimpleVAE(input_dim=input_dim, hidden_dim=128, latent_dim=20)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 50
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_data in dataloader:
            x = batch_data[0]
            
            # 前向传播
            x_recon, mu, log_var = model(x)
            
            # 计算损失
            loss = vae_loss_function(x_recon, x, mu, log_var)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # 生成一些样本
    model.eval()
    with torch.no_grad():
        # 从潜在空间中随机采样
        z = torch.randn(16, 20)  # 16个样本，潜在维度为20
        
        # 解码生成样本
        samples = model.decode(z)
        
        # 可视化生成的样本
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        
        for i, ax in enumerate(axes.flat):
            # 重塑为8x8图像
            img = samples[i].view(8, 8).numpy()
            
            # 显示图像
            ax.imshow(img, cmap='gray')
            ax.set_title(f'样本 {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 可视化潜在空间
    model.eval()
    with torch.no_grad():
        # 获取所有数据的潜在表示
        mu_list = []
        labels_list = []
        
        for i in range(0, len(data), 32):
            batch = data[i:i+32]
            mu, _ = model.encode(batch)
            mu_list.append(mu)
            labels_list.extend([j % 10 for j in range(i, min(i+32, len(data)))])
        
        mu_all = torch.cat(mu_list, dim=0)
        
        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        mu_2d = tsne.fit_transform(mu_all.numpy())
        
        # 可视化潜在空间
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='数字类型')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.title('VAE潜在空间可视化')
        plt.grid(True)
        plt.show()
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.grid(True)
    plt.show()
    
    return model

# ==============================
# 11. 主程序
# ==============================

def main():
    """
    主程序，运行所有示例
    """
    print("神经网络基础知识示例程序")
    print("=" * 50)
    
    # 神经网络结构示例
    example_neural_network_structure()
    
    # 激活函数可视化
    visualize_activation_functions()
    
    # 损失函数可视化
    visualize_loss_functions()
    
    # 优化算法比较
    compare_optimizers()
    
    # 前馈神经网络分类任务
    classification_model = train_feedforward_nn_classification()
    
    # 前馈神经网络回归任务
    regression_model = train_feedforward_nn_regression()
    
    # CNN特征图可视化
    visualize_cnn_feature_maps()
    
    # RNN序列预测
    rnn_model = example_rnn_sequence_prediction()
    
    # 注意力权重可视化
    visualize_attention_weights()
    
    # GNN节点分类
    gnn_model = example_gnn_node_classification()
    
    # VAE生成模型
    vae_model = example_vae_mnist_generation()
    
    print("\n所有示例运行完成！")

if __name__ == "__main__":
    main()
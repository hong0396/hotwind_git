import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# 创建数据
X = np.linspace(-1, 1, 200)
# 数据随机化
np.random.shuffle(X)
# 创建数据及参数, 并加入噪声
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))

# 绘制数据
plt.scatter(X, Y)
plt.show()

# 分为训练数据和测试数据
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# 使用keras创建神经网络
# Sequential是指一层层堆叠的神经网络
# Dense是指全连接层
# 定义model
model = Sequential()
# 定义第一层, 由于是回归模型, 因此只有一层
model.add(Dense(units = 1, input_dim = 1))

# 选择损失函数和优化方法
model.compile(loss = 'mse', optimizer = 'sgd')

print '----Training----'
# 训练过程
for step in range(501):
    # 进行训练, 返回损失(代价)函数
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print 'loss: ', cost

print '----Testing----'    
# 训练结束进行测试
cost = model.evaluate(X_test, Y_test, batch_size = 40)
print 'test loss: ', cost

# 获取参数
W, b = model.layers[0].get_weights()
print 'Weights: ',W
print 'Biases: ', b

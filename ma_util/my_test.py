import tensorflow as tf
import numpy as np
import os

# 例子1，拟合y_data的函数，权重和偏置分别趋近0.1和0.3

# np.random.rand(100)生成100个[0,1]之间的随机数，构成1维数组
# np.random.rand(2,3)生成2行3列的二维数组
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 权重偏置这些不断更新的值用tf变量存储，
# tf.random_uniform()的参数意义：(shape,min,max)
# 偏置初始化为0
weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

# 例子2，构建一个神经网络

# 添加神经层的函数，它有四个参数：输入值、输入的形状、输出的形状和激励函数，
# Wx_plus_b是未激活的值，函数返回激活值。
def add_layer(inputs, in_size, out_size,  activation_function=None):
    # tf.random_normal()参数为shape，还可以指定均值和标准差
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 构建训练数据
# np.linspace()在-1和1之间等差生成300个数字
# noise是正态分布的噪声，前两个参数是正态分布的参数，然后是size
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05,  x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需的神经网络的输入。
# 第二个参数为shape：None代表行数不定，1是列数。
# 这里的行数就是样本数，列数是每个样本的特征数。
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 输入层1个神经元（因为只有一个特征），隐藏层10个，输出层1个。
# 调用函数定义隐藏层和输出层，输入size是上一层的神经元个数（全连接），输出size是本层个数。
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均作为损失函数。
# reduction_indices表示最后数据的压缩维度，好像一般不用这个参数（即降到0维，一个标量）。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step =  tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化变量，激活，执行运算
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        # training
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
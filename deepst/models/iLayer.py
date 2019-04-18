from keras import backend as K
from keras.engine.topology import Layer
# from keras.layers import Dense
import numpy as np
class iLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)
    #建立一个层， 模拟wx的式子开始创建变量参与到计算中，以w作为层内的链接，
    def build(self, input_shape):
        #print('我被调用了吗？？？')
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]
    #让W和x进行计算，这是融合时的需要，可以将不同类型的时间特性数据合在一起。
    def call(self, x, mask=None):
        #print('我被调用了吗？？')    #被调用了，说明分配权重了。
        return x * self.W
    def get_output_shape_for(self, input_shape):
        return input_shape

import math
from keras.layers import Dense
from keras.layers import merge as Merge
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf


def my_init_sigmoid(shape, dtype=None):
    rnd = K.random_uniform(
        shape, 0., 1., dtype)
    from keras.initializers import _compute_fans
    fan_in, fan_out = _compute_fans(shape)
    return 8. * (rnd - 0.5) * math.sqrt(6) / math.sqrt(fan_in + fan_out)

def my_init_others(shape, dtype=None):
    rnd = K.random_uniform(
        shape, 0., 1., dtype)
    from keras.initializers import _compute_fans
    fan_in, fan_out = _compute_fans(shape)
    return 2. * (rnd - 0.5) / math.sqrt(fan_in)

class DeepCCA():
    def __init__(self, layer_sizes1,
                 layer_sizes2, layer_sizes3, layer_sizes4,input_size1,
                 input_size2, outdim_size, reg_par, use_all_singular_values):

        self.layer_sizes1 = layer_sizes1  # [1024, 1024, 1024, outdim_size]
        self.layer_sizes2 = layer_sizes2  # [1024, 1024, 1024, outdim_size]
        self.layer_sizes3 = layer_sizes3
        self.layer_sizes4 = layer_sizes4
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.outdim_size = outdim_size
#      MLP
        self.input_view1 = tf.placeholder(tf.float32, [None, input_size1])
        self.input_view2 = tf.placeholder(tf.float32, [None, input_size2])
        # tf Graph input cnn
#        self.input_view1 = tf.placeholder("float", [None, 256])
#        self.input_view2 = tf.placeholder("float", [None, 256])
        self.label1 = tf.placeholder("float", [None, 52])
        self.label2 = tf.placeholder("float", [None, 52])
        
#        # Define weights
#        self.weights = {
#            # 5x5 conv, 1 input, 32 outputs
#            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
#            # 5x5 conv, 32 inputs, 64 outputs
#            'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
#            # fully connected, 7*7*64 inputs, 1024 outputs
#            'wd1': tf.Variable(tf.random_normal([4*256, 1024])),
#            # 1024 inputs, 10 outputs (class prediction)
#            'cnnout': tf.Variable(tf.random_normal([4*4*64, 52])),
#            'mlpout': tf.Variable(tf.random_normal([52, 52]))
#        }
#        self.biases = {
#            'bc1': tf.Variable(tf.random_normal([32])),
#            'bc2': tf.Variable(tf.random_normal([64])),
#            'bd1': tf.Variable(tf.random_normal([1024])),
#            'cnnout': tf.Variable(tf.random_normal([52]))
#        }

#       neural network mlp
        self.output_view1= self.build_mlp_net(self.input_view1, layer_sizes1, reg_par)
        self.output_view2= self.build_mlp_net(self.input_view2, layer_sizes2, reg_par)
#        classification
        self.output_view1_class = self.build_mlp_net(self.input_view1, layer_sizes3, reg_par)
        self.output_view2_class = self.build_mlp_net(self.input_view2, layer_sizes4, reg_par)
#       neural network CNN        
#        self.output_view1 = self.conv_net(self.input_view1,self.weights,self.biases)
#        self.output_view2 = self.conv_net(self.input_view2,self.weights,self.biases)
        self.neg_corr,self.value = self.neg_correlation(self.output_view1, self.output_view2, use_all_singular_values)
        
        
#     CNN
	# Create some wrappers for simplicity
    def conv2d(self,x, W, b, strides=1):
	    # Conv2D wrapper, with bias and relu activation
	    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	    x = tf.nn.bias_add(x, b)
	    return tf.nn.relu(x)

    def maxpool2d(self,x, k=2):
	    # MaxPool2D wrapper
	    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
			          padding='SAME')

#	# Create model
#    def conv_net(self,x, weights, biases, dropout=0.5):
#	    # Reshape input picture
#        x = tf.reshape(x, shape=[-1, 16, 16, 1])
#	    # Convolution Layer
#        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
#	    # Max Pooling (down-sampling)
#        conv1 = self.maxpool2d(conv1, k=2)
#	    # Convolution Layer
#        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
#	    # Max Pooling (down-sampling)
#        conv2 = self.maxpool2d(conv2, k=2)
#	    # Fully connected layer
#	    # Reshape conv2 output to fit fully connected layer input
#        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
#        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#        fc1 = tf.nn.relu(fc1)
#        # Apply Dropout
#        fc1 = tf.nn.dropout(fc1, dropout)
##        conv2 = tf.reshape(conv2, [-1, 1024])
#        out = tf.add(tf.matmul(fc1, weights['cnnout']), biases['cnnout'])
#        return out
    
    
#   mlp
    def build_mlp_net(self, input, layer_sizes, reg_par):
        output = input
        for l_id, ls in enumerate(layer_sizes):
            if l_id == len(layer_sizes) - 1:
                activation = None
                kernel_initializer = my_init_others
            else:
                activation = tf.nn.sigmoid
                kernel_initializer = my_init_sigmoid

            output = Dense(ls, activation=activation,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2(reg_par))(output)
            
        return output
    
#    loss computition
    def neg_correlation(self, output1, output2, use_all_singular_values):
        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-12

        # unpack (separate) the output of networks for view 1 and view 2
        H1 = tf.transpose(output1)
        H2 = tf.transpose(output2)
       
        m = tf.shape(H1)[1]
        H1bar = H1 - (1.0 / tf.cast(m, tf.float32)) * tf.matmul(H1, tf.ones([m, m]))
        H2bar = H2 - (1.0 / tf.cast(m, tf.float32)) * tf.matmul(H2, tf.ones([m, m]))
        SigmaHat12 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H1bar, tf.transpose(H2bar))
        SigmaHat11 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H1bar, tf.transpose(H1bar)) + r1 * tf.eye(self.outdim_size)
        SigmaHat22 = (1.0 / (tf.cast(m, tf.float32) - 1)) * tf.matmul(H2bar, tf.transpose(H2bar)) + r2 * tf.eye(self.outdim_size)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = tf.linalg.eigh(SigmaHat11)
        [D2, V2] = tf.linalg.eigh(SigmaHat22)

        # Added to increase stability
        posInd1 = tf.where(tf.greater(D1, eps))
        posInd1 = tf.reshape(posInd1, [-1, tf.shape(posInd1)[0]])[0]
        D1 = tf.gather(D1, posInd1)
        V1 = tf.gather(V1, posInd1)

        posInd2 = tf.where(tf.greater(D2, eps))
        posInd2 = tf.reshape(posInd2, [-1, tf.shape(posInd2)[0]])[0]
        D2 = tf.gather(D2, posInd2)
        V2 = tf.gather(V2, posInd2)

        SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.linalg.diag(D1 ** -0.5)), tf.transpose(V1))
        SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.linalg.diag(D2 ** -0.5)), tf.transpose(V2))
        print(SigmaHat22RootInv.shape)
        Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
        value=None
        if use_all_singular_values:
            # all singular values are used to calculate the correlation
            # corr = tf.sqrt(tf.linalg.trace(tf.matmul(tf.transpose(Tval), Tval)))  ### The usage of "sqrt" here is wrong!!!
            Tval.set_shape([self.outdim_size, self.outdim_size])
            s = tf.linalg.svd(Tval, compute_uv=False)
            value=s
            corr = tf.reduce_sum(s)
        else:
            # just the top outdim_size singular values are used
            [U, V] = tf.linalg.eigh(tf.matmul(tf.transpose(Tval), Tval))
            non_critical_indexes = tf.where(tf.greater(U, eps))
            non_critical_indexes = tf.reshape(non_critical_indexes, [-1, tf.shape(non_critical_indexes)[0]])[0]
            U = tf.gather(U, non_critical_indexes)
            U = tf.gather(U, tf.nn.top_k(U[:, ]).indices)
            value=tf.sqrt(U[0:self.outdim_size])
            corr = tf.reduce_sum(tf.sqrt(U[0:self.outdim_size]))
        return -corr,value


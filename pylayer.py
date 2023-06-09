from turtle import forward
import numpy as np
from math import *

'''
    LINEAR
    Implementation of the linear layer (also called fully connected layer)
    which performs linear transoformation on input data y = xW + b.
    This layer has two learnable parameters, weight of shape (input_channel, output_channel)
    and bias of shape (output_channel), which are specified and initalized in init_param()
    function. In this assignment, you need to implement both forward and backward computation
    Arguments:
        input_channel  -- integer, number of input channels
        output_channel -- integer, number of output channels
'''

class Linear(object):

    def __init__(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.init_param()

    def init_param(self):
        self.weight = (np.random.randn(self.input_channel,self.output_channel) * sqrt(2.0/(self.input_channel+self.output_channel))).astype(np.float32)
        self.bias = np.zeros((self.output_channel))

    '''
        Forward computation of linear layer, you may want to save some intermediate
        variable to class membership (self.) for reusing in backward computation.
        Arguments:
            input -- numpy array of shape (N, input_channel)

        Output:
            output -- numpy array of shape (N, output_channel)
    '''
    def forward(self, input):
        self.input = input
        ##################################################
        # TODO: YOUR CODE HERE: forward
        ##################################################
        # output = np.dot(self.input, self.weight)
        output = np.einsum('Ni,io -> No', self.input, self.weight)
        return output + self.bias

    '''
        Backward computation of linear layer, you need to compute the gradient
        w.r.t input, weight and bias respectively. You need to reuse the variable in forward
        computation to compute backward gradient.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel)

        Output:
            grad_input -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_weight -- numpy array of shape (input_channel, output_channel), gradient w.r.t weight
            grad_bias --  numpy array of shape (output_channel), gradient w.r.t bias
    '''
    def backward(self, grad_output):
        ##################################################
        # TODO: YOUR CODE HERE: backward
        ##################################################        
        grad_bias = np.einsum('No -> o', grad_output)
        grad_weight = np.einsum('Ni,No -> io', self.input, grad_output)
        grad_input = np.einsum('io,No -> Ni', self.weight, grad_output)
        return grad_input, grad_weight, grad_bias

'''
    BatchNorm1D
    DO NOT USE
    Implementation of batch normalization (or BN) layer, which performs normalization and rescaling
    on input data. Specifically, for input data X of shape (N,input_channel), BN layers firstly normalized the data along batch dimension
    by the mean E(x), variance Var(X) that are computed within batch data and both have shape of (input_channel)
    Then BN re-scales the normalized data with learnable parameters beta and gamma, both have shape of (input_channel).
    So the forward formula is written as

            Y = ((X - mean(X)) /  sqrt(Var(x) + eps)) * gamma + beta

    At the same time, BN layer maintains a running_mean and running_variance that are momentumly updated during
    forward iteration and would replace batch-wise E(x) and Var(x) for testing. The equations are:

            running_mean = (1 - momentum) * E(x)   +  momentum * running_mean
            running_var =  (1 - momentum) * Var(x) +  momentum * running_var

    During test time, since the batch size could be an arbitrary number, the statistic inside batch may not be a good approximation of data distribution,
    thus we need instead using running_mean and running_var to perform normalization.
    Thus the forward formular is modified to:

            Y = ((X - running_mean) /  sqrt(running_var + eps)) * gamma + beta

    Overall, BN maintains 4 learnable parameters with shape of (input_channel),
    running_mean, running_var, beta and gamma.  In this assignment, you need
    to complete the forward and backward computation and handle the case for

    Arguments:
        input_channel -- integer, number of input channel
        momentum      -- float,   the momentum value used for the running_mean and running_var computation
'''
class BatchNorm1d(object):

    def __init__(self, input_channel, momentum = 0.9):
        self.input_channel = input_channel
        self.momentum = momentum
        self.eps = 1e-3
        self.init_param()

    def init_param(self):
        self.r_mean = np.zeros((self.input_channel)).astype(np.float32)
        self.r_var = np.ones((self.input_channel)).astype(np.float32)
        self.beta = np.zeros((self.input_channel)).astype(np.float32)
        self.gamma = (np.random.rand(self.input_channel) * sqrt(2.0/(self.input_channel))).astype(np.float32)
    '''
        Forward computation of batch normalization layer and momentumly updated the running mean and running variance
        You may want to save some intermediate variables to class membership (self.) and you should take care of different behaviors
        during training and testing.

        Arguments:
            input -- numpy array (N, input_channel)
            train -- bool, boolean indicator to specify the running mode, True for training and False for testing
    '''
    def forward(self, input, train):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        self.input = input
        if train:
            mu = np.mean(input, axis =0)
            var = np.var(input, axis =0)
            self.mu = mu
            self.var = var
            self.r_mean = self.r_mean * self.momentum + (1 - self.momentum) * mu
            self.r_var = self.r_var * self.momentum + (1 - self.momentum) * var
            self.input_norm = (input - mu[None,:]) / np.sqrt(var[None,:] + self.eps)
            output = (self.input_norm * self.gamma) + self.beta
        else:
            input_norm = (input - self.r_mean[None,:])/np.sqrt(self.r_var[None,:] + self.eps)
            output = (input_norm * self.gamma[None,:]) + self.beta[None,:]
        return output
    '''
        Backward computationg of batch normalization layer
        You need to write gradient w.r.t input data, gamma and beta
        It's recommend to follow the chain rule to firstly compute the gradient w.r.t to intermediate variable to
        simplify the computation.

        Arguments:
            grad_output -- numpy array of shape (N, input_channel)

        Output:
            grad_input -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_gamma -- numpy array of shape (input_channel), gradient w.r.t gamma
            grad_beta  -- numpy array of shape (input_channel), gradient w.r.t beta
    '''
    def backward(self, grad_output):
        ##########################################################################
        # TODO: YOUR CODE HERE
        ##########################################################################
        N = grad_output.shape[0]
        dxdhat = self.gamma[None,:] * grad_output
        output_term1 =  (1./N) * 1./np.sqrt(self.var + self.eps)
        output_term2 = N * dxdhat
        output_term3 = np.sum(dxdhat, axis=0)
        output_term4 = self.input_norm * np.sum(dxdhat * self.input_norm, axis=0)
        grad_input = output_term1 * (output_term2 - output_term3 - output_term4)
        grad_gamma = np.sum(grad_output * self.input_norm, axis = 0)
        grad_beta = np.sum(grad_output, axis = 0)
        return grad_input, grad_gamma, grad_beta

'''
    RELU
    Implementation of relu (rectified linear unit) layer. Relu is the no-linear activating function that
    set all negative values to zero and the formua is y = max(x,0).
    This layer has no learnable parameters and you need to implement both forward and backward computation
    Arguments:
        None
'''

class ReLU(object):
    def __init__(self):
        pass
    '''
        Forward computation of relu and you may want to save some intermediate variables to class membership (self.)
        Arguments:
            input -- numpy array of arbitrary shape

        Output:
            output -- numpy array having the same shape as input.
    '''
    def forward(self, input):
        self.input = input
        return np.maximum(input, 0)

    '''
        Backward computation of relu, you can either in-place modify the grad_output or create a copy.
        Arguments:
            grad_output-- numpy array having the same shape as input

        Output:
            grad_input -- numpy array has the same shape as grad_output. gradient w.r.t input
    '''
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input<0] = 0
        return grad_input
'''
    CROSS_ENTROPY_LOSS_WITH_SOFTMAX
    Implementation of the combination of softmax function and cross entropy loss.
    In classification task, we usually firstly apply softmax to map class-wise prediciton
    into the probabiltiy distribution then we use cross entropy loss to maximise the likelihood
    of ground truth class's prediction. Since softmax includes exponential term and cross entropy includes
    log term, we can simplify the formula by combining these two functions togther so that log and exp term could cancell out
    mathmatically and we can avoid precision lost with float point numerical computation.
    If we ignore the index on batch sizel and assume there is only one grouth truth per sample,
    the formula for softmax and cross entropy loss are:
        Softmax: prob[i] = exp(x[i]) / \sum_{j}exp(x[j])
        Cross_entropy_loss:  - 1 * log(prob[gt_class])
    Combining these two function togther, we got
        cross_entropy_with_softmax: -x[gt_class] + log(\sum_{j}exp(x[j]))
    In this assignment, you will implement both forward and backward computation.
    Arguments:
        None
'''
class CrossEntropyLossWithSoftmax(object):
    def __init__(self):
        pass
    '''
        Forward computation of cross entropy with softmax, you may want to save some intermediate variables to class membership (self.)
        Arguments:
            input    -- numpy array of shape (N, C), the prediction for each class, where C is number of class
            gt_label -- numpy array of shape (N), it's a integer array and the value range from 0 to C-1 which
                        specify the ground truth class for each input
        Output:
            output   -- numpy array of shape (N), containing the cross entropy loss on each input
    '''
    def forward(self, input, gt_label):
        t = np.max(input,axis=1).reshape(input.shape[0],1)
        input = input - t
        exp = np.exp(input)
        self.gt_label = gt_label
        self.prob = exp / np.sum(exp, axis = -1)[:,None]
        log_term = np.log(np.sum(exp, axis = -1))
        output = -input[np.arange(input.shape[0]), gt_label] + log_term
        return output

    '''
        Backward computation of cross entropy with softmax. It's recommended to resue the variable
        in forward computation to simplify the formula.
        Arguments:
            grad_output -- numpy array of shape (N)

        Output:
            output   -- numpy array of shape (N, C), the gradient w.r.t input of forward function
    '''
    def backward(self, grad_output):
        self.prob[np.arange(self.prob.shape[0]),self.gt_label] -= 1
        return grad_output[:,None] * self.prob

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]

class Conv2d:
    def __init__(self, input_channel, output_channel, kernel_size, padding = 0, stride = 1):
        self.output_channel = output_channel
        self.input_channel = input_channel
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.pad = padding
        self.stride = stride
        self.init_param()

    def init_param(self):
        self.weight = (np.random.randn(self.output_channel, self.input_channel, self.kernel_h, self.kernel_w) * sqrt(2.0/(self.input_channel + self.output_channel))).astype(np.float32)
        self.bias = np.zeros(self.output_channel).astype(np.float32)

    def forward(self, x):
        FN, C, FH, FW = self.weight.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.weight.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.bias
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.weight.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        db = np.sum(dout, axis=0)

        dW = np.dot(self.col.T, dout)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)
        
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx, dW, db

class MaxPool2d:
    def __init__(self, kernel_size, padding = 0, stride = 1):
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.pad = padding
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.kernel_h) / self.stride)
        out_w = int(1 + (W - self.kernel_w) / self.stride)

        col = im2col(x, self.kernel_h, self.kernel_w, self.stride, self.pad)
        col = col.reshape(-1, self.kernel_h * self.kernel_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        print(x.shape,out.shape)

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.kernel_h * self.kernel_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.kernel_h, self.kernel_w, self.stride, self.pad)

        return dx

##########################################################################
# MY EXTRA CODE HERE
##########################################################################
class Flatten(object):
    '''
        Used in convnets, when output images are needed to be flattened.
        Do it by memorize the shape when forwarding.

        input tensor: (N, any...)
        output tensor: (N, M)
    '''
    def __init__(self):
        pass
    def forward(self, input):
        self.inshape = input.shape
        return input.reshape(self.inshape[0], -1)
    def backward(self, grad_input):
        return grad_input.reshape(self.inshape)

class BatchNorm2d(BatchNorm1d):
    def __init__(self, input_channel, momentum = 0.9):
        super(BatchNorm2d, self).__init__(input_channel, momentum)
    @staticmethod
    def to_inner(tensor):
        tensor = tensor.transpose(0, 2, 3, 1)
        tensor = tensor.reshape(-1, tensor.shape[-1])
        # tensor is NHW, C
        return tensor
    def to_outer(self, tensor):
        # tensor is NHW, C
        N, C, H, W = self.in_shape
        tensor = tensor.reshape(N, H, W, C)
        tensor = tensor.transpose(0, 3, 1, 2)
        return tensor
    def forward(self, input, train):
        self.in_shape = input.shape
        output = super(BatchNorm2d, self).forward(self.to_inner(input), train)
        return self.to_outer(output)
    def backward(self, grad_output):
        grad_input, grad_gamma, grad_beta = super(BatchNorm2d, self).backward(self.to_inner(grad_output))
        return self.to_outer(grad_input), grad_gamma, grad_beta

class BasicBlock(object):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Layers
        self.conv1 = Conv2d(in_channels, out_channels, (3, 3), padding=1, stride=stride)#, bias=False)
        # self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, (3, 3), padding=1, stride=1)#, bias=False)
        # self.bn2 = BatchNorm2d(out_channels)
        self.relu1, self.relu2 = ReLU(), ReLU()
        self.downsample = downsample

        self.params_ref = [
            self.conv1.weight, self.conv1.bias, self.bn1.gamma, self.bn1.beta,
            self.conv2.weight, self.conv2.bias, self.bn2.gamma, self.bn2.beta,
        ]
        if self.downsample:
            self.params_ref += [
                self.downsample[0].weight, self.downsample[0].bias,
                self.downsample[1].gamma, self.downsample[1].beta,
            ]

    def forward(self, input, train_mode=True):
        x = input
        residual = self.conv1.forward(input)
        residual = self.bn1.forward(residual, train_mode)
        residual = self.relu1.forward(residual)
        residual = self.conv2.forward(residual)
        residual = self.bn2.forward(residual, train_mode)
        if self.downsample:
            x = self.downsample[0].forward(x)
            x = self.downsample[1].forward(x, train_mode)
        x += residual
        x = self.relu2.forward(x)
        return x
    
    def backward(self, grad_output):
        grad_output = self.relu2.backward(grad_output)
        grad_x = grad_output
        grad_res = grad_output
        if self.downsample:
            grad_x, gw_bn_ds, gb_bn_ds = self.downsample[1].backward(grad_x)
            grad_x, gw_conv_ds, gb_conv_ds = self.downsample[0].backward(grad_x)
        grad_res, gw_bn2, gb_bn2 = self.bn2.backward(grad_res)
        grad_res, gw_conv2, gb_conv2 = self.conv2.backward(grad_res)
        grad_res = self.relu1.backward(grad_res)
        grad_res, gw_bn1, gb_bn1 = self.bn1.backward(grad_res)
        grad_res, gw_conv1, gb_conv1 = self.conv1.backward(grad_res)
        grad_input = grad_x + grad_res
        output_grads = [
            gw_conv1, gb_conv1, gw_bn1, gb_bn1,
            gw_conv2, gb_conv2, gw_bn2, gb_bn2
        ]
        if self.downsample:
            output_grads += [gw_conv_ds, gb_conv_ds, gw_bn_ds, gb_bn_ds]
        return grad_input, *output_grads

class BottleNeck(object):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        # layers
        self.conv1 = Conv2d(in_channels, out_channels, (1, 1))#, bias=False)
        # self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, (3, 3), padding=1, stride=stride)#, bias=False)
        # self.bn2 = BatchNorm2d(out_channels)
        self.conv3 = Conv2d(out_channels, out_channels*self.expansion, (1, 1))#, bias=False)
        # self.bn3 = BatchNorm2d(out_channels*self.expansion)
        self.relu1, self.relu2, self.relu3 = ReLU(), ReLU(), ReLU()
        self.downsample = downsample

        self.params_ref = [
            self.conv1.weight, self.conv1.bias, self.bn1.gamma, self.bn1.beta,
            self.conv2.weight, self.conv2.bias, self.bn2.gamma, self.bn2.beta,
            self.conv3.weight, self.conv3.bias, self.bn3.gamma, self.bn3.beta,
        ]
        if self.downsample:
            self.params_ref += [
                self.downsample[0].weight, self.downsample[0].bias,
                self.downsample[1].gamma, self.downsample[1].beta,
            ]

    def forward(self, input, train_mode=True):
        x = input
        residual = self.conv1.forward(input)
        residual = self.bn1.forward(residual)
        residual = self.relu1.forward(residual)
        residual = self.conv2.forward(residual)
        residual = self.bn2.forward(residual)
        residual = self.relu2.forward(residual)
        residual = self.conv3.forward(residual)
        residual = self.bn3.forward(residual)
        if self.downsample:
            x = self.downsample[0].forward(x)
            x = self.downsample[1].forward(x, train_mode)
        x += residual
        x = self.relu3.forward(x)
        return x
    
    def backward(self, grad_output):
        grad_output = self.relu3.backward(grad_output)
        grad_x = grad_output
        grad_res = grad_output
        if self.downsample:
            grad_x, gw_bn_ds, gb_bn_ds = self.downsample[1].backward(grad_x)
            grad_x, gw_conv_ds, gb_conv_ds = self.downsample[0].backward(grad_x)
        grad_res, gw_bn3, gb_bn3 = self.bn3.backward(grad_res)
        grad_res, gw_conv3, gb_conv3 = self.conv3.backward(grad_res)
        grad_res = self.relu2.backward(grad_res)
        grad_res, gw_bn2, gb_bn2 = self.bn2.backward(grad_res)
        grad_res, gw_conv2, gb_conv2 = self.conv2.backward(grad_res)
        grad_res = self.relu1.backward(grad_res)
        grad_res, gw_bn1, gb_bn1 = self.bn1.backward(grad_res)
        grad_res, gw_conv1, gb_conv1 = self.conv1.backward(grad_res)
        grad_input = grad_x + grad_res
        output_grads = [
            gw_conv1, gb_conv1, gw_bn1, gb_bn1,
            gw_conv2, gb_conv2, gw_bn2, gb_bn2,
            gw_conv3, gb_conv3, gw_bn3, gb_bn3
        ]
        if self.downsample:
            output_grads += [gw_conv_ds, gb_conv_ds, gw_bn_ds, gb_bn_ds]
        return grad_input, *output_grads
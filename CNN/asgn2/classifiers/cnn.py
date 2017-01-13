import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    mu, sigma = 0, weight_scale
    C,H,W = input_dim
    F = num_filters
    HH = filter_size
    WW = filter_size
    op = (H/2)*(W/2)*F
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    self.params['W1'] = np.random.normal(mu, sigma, (F, C, HH, WW))
    self.params['b1'] = np.zeros(F)
    self.params['W2'] = np.random.normal(mu, sigma, (op,hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(mu, sigma, (hidden_dim,num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    conv_out, conv_cache = conv_forward_fast(X, self.params['W1'], self.params['b1'], conv_param)
    conv_relu_out, conv_relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = max_pool_forward_fast(conv_relu_out, pool_param)
    af_relu_out, af_relu_cache = affine_relu_forward(pool_out, self.params['W2'], self.params['b2'])
    scores, fc_cache = affine_forward(af_relu_out, self.params['W3'], self.params['b3'])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2']) + np.sum(self.params['W3']*self.params['W3'])) 

    dout, grads['W3'], grads['b3'] = affine_backward(dout, fc_cache)
    dout, grads['W2'], grads['b2']= affine_relu_backward(dout, af_relu_cache)
    dout = max_pool_backward_fast(dout, pool_cache)
    dout = relu_backward(dout, conv_relu_cache)
    dout, grads['W1'], grads['b1'] = conv_backward_fast(dout, conv_cache)

    grads['W1'] += self.reg * self.params['W1']
    grads['W2'] += self.reg * self.params['W2']
    grads['W3'] += self.reg * self.params['W3']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class ExperimentConvNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7, num_classes=10, 
               hidden_dim=100, dropout=0, use_batchnorm=False, use_xavier=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.dtype = dtype
    self.params = {}
    self.use_xavier = use_xavier

    mu, sigma = 0, weight_scale
    C,H,W = input_dim
    F = num_filters
    HH = filter_size
    WW = filter_size
    op = (H/2)*(W/2)*F
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    if self.use_xavier is True:
        self.params['W1'] = np.random.normal(mu, sigma, (F, C, HH, WW))/np.sqrt(C/2)
    else:
        self.params['W1'] = np.random.normal(mu, sigma, (F, C, HH, WW))
    self.params['b1'] = np.zeros(F)
    if self.use_xavier is True:
        self.params['W2'] = np.random.normal(mu, sigma, (op,hidden_dim))/np.sqrt(op/2)
    else:
        self.params['W2'] = np.random.normal(mu, sigma, (op,hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    if self.use_xavier is True:
        self.params['W3'] = np.random.normal(mu, sigma, (hidden_dim,num_classes))/np.sqrt(hidden_dim/2)
    else:
        self.params['W3'] = np.random.normal(mu, sigma, (hidden_dim,num_classes))
    self.params['b3'] = np.zeros(num_classes)

    if self.use_batchnorm is not False:
      self.params['gamma1'], self.params['beta1'] = np.ones(self.params['W1'].shape[0]), np.zeros(self.params['W1'].shape[0])
      self.params['gamma2'], self.params['beta2'] = np.ones(self.params['W2'].shape[1]), np.zeros(self.params['W2'].shape[1])
   
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    conv - [batch norm] - relu - 2x2 max pool - affine -[batch norm] - relu - affine - softmax
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    mode = 'test' if y is None else 'train'
    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None

    #[Dropout] - conv - [BN] - relu - 2x2 max pool -[Dropout]- affine -[BN]- relu - [Dropout] - affine - softmax
    if self.use_dropout:
      do_out, do1_cache = dropout_forward(X, self.dropout_param)
    else:
      do_out = X
    conv_out, conv_cache = conv_forward_fast(do_out, self.params['W1'], self.params['b1'], conv_param)
    if self.use_batchnorm is not False:
        bn1_out, bn1_cache = spatial_batchnorm_forward(conv_out, self.params['gamma1'], self.params['beta1'], self.bn_params[0])
    else:
        bn1_out = conv_out
    conv_relu_out, conv_relu_cache = relu_forward(bn1_out)
    pool_out, pool_cache = max_pool_forward_fast(conv_relu_out, pool_param)
    if self.use_dropout:
      do_out, do2_cache = dropout_forward(pool_out, self.dropout_param)
    else:
      do_out = pool_out
    af_out, af_cache = affine_forward(do_out, self.params['W2'], self.params['b2'])
    if self.use_batchnorm is not False:
        bn2_out, bn2_cache = batchnorm_forward(af_out, self.params['gamma2'], self.params['beta2'], self.bn_params[1])
    else:
        bn2_out = af_out
    af_relu_out, af_relu_cache = relu_forward(bn2_out)  
    if self.use_dropout:
      do_out, do3_cache = dropout_forward(af_relu_out, self.dropout_param)
    else:
      do_out = af_relu_out
    scores, fc_cache = affine_forward(do_out, self.params['W3'], self.params['b3'])


    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)

    loss += 0.5 * self.reg * (np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2']) + np.sum(self.params['W3']*self.params['W3'])) 

    #[Dropout] - conv - [BN] - relu - 2x2 max pool -[Dropout]- affine -[BN]- relu - [Dropout] - affine - softmax
    dout, grads['W3'], grads['b3'] = affine_backward(dout, fc_cache)
    if self.use_dropout:
        dout = dropout_backward(dout, do3_cache)
    dout = relu_backward(dout, af_relu_cache)
    if self.use_batchnorm is not False:
        dout, grads['gamma2'], grads['beta2'] = batchnorm_backward(dout, bn2_cache)
    dout, grads['W2'], grads['b2'] = affine_backward(dout, af_cache)
    if self.use_dropout:
        dout = dropout_backward(dout, do2_cache)
    dout = max_pool_backward_fast(dout, pool_cache)
    dout = relu_backward(dout, conv_relu_cache)
    if self.use_batchnorm is not False:
        dout, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dout, bn1_cache)
    dout, grads['W1'], grads['b1'] = conv_backward_fast(dout, conv_cache)
    if self.use_dropout:
        dout = dropout_backward(dout, do1_cache)

    grads['W1'] += self.reg * self.params['W1']
    grads['W2'] += self.reg * self.params['W2']
    grads['W3'] += self.reg * self.params['W3']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


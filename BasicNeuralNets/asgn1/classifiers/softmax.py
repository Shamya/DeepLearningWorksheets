import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
    #negative log normalized exponentiated score
    scores = X[i].dot(W)
    scores -= np.max(scores)  #for numerical stability
    scores = np.exp(scores)
    scores = scores /np.sum(scores)
    fyi = scores[y[i]]
    loss += -np.log(fyi)
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] += X[i] * (scores[j] - 1)
      else:
        dW[:,j] += X[i] * scores[j]

  #take mean
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #negative log normalized exponentiated score 
  scores = X.dot(W)
  scores -= np.max(scores)
  scores = np.exp(scores) 
  scores = scores / np.reshape(np.sum(scores, axis = 1), (scores.shape[0],1))
  loss = np.sum(-np.log(scores[(np.arange(scores.shape[0]),y)]))

  indicator = np.zeros_like(scores)
  indicator[np.arange(indicator.shape[0]), y] = -1
  scores = scores + indicator
  dW = X.T.dot(scores)


  #take mean
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


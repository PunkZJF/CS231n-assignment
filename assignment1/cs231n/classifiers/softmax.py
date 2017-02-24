# -*- coding: utf-8 -*-

'''
created by Jesse Zhang
date: 2016-11-25
'''

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
  num_train, dim = X.shape
  num_classes = W.shape[1]
  dW_each = np.zeros_like(W)
  f = X.dot(W)
  f_max = np.reshape(np.max(f, axis=1), (num_train, 1))
  prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
  y_trueclass = np.zeros_like(prob)
  y_trueclass[np.arange(num_train), y] = 1.0
  for i in xrange(num_train):
    for j in xrange(num_classes):
      #损失函数的公式L = -(1/N)∑i∑j1(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)
      loss += -(y_trueclass[i, j] * np.log(prob[i, j]))
      #梯度的公式 ∇Wk L = -(1/N)∑i xiT(pi,m-Pm) + 2λWk,where Pk = exp(fk)/∑j exp(fj)
      dW_each[:, j] = -(y_trueclass[i, j] - prob[i, j]) * X[i, :]
    dW += dW_each
  loss /= num_train
  dW /= num_train
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  f = X.dot(W)
  f_max = np.reshape(np.max(f, axis=1), (num_train, 1))
  prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
  y_trueclass = np.zeros_like(prob)
  y_trueclass[np.arange(num_train), y] = 1.0
  loss += -np.sum(y_trueclass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
  dW += -np.dot(X.T, y_trueclass - prob) / num_train + reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


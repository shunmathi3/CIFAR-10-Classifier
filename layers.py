import numpy as np


def fc_forward(x, w, b):
#computes forward pass for a fully connected layer
  out = x.dot(w) + b  # Matrix multiplication followed by bias addition
  cache = (x, w, b)
  return out, cache

def fc_backward(grad_out, cache):
  #computes backward pass for a fully connected layer
  x, w, b = cache
  grad_x = grad_out.dot(w.T)  
  grad_w = x.T.dot(grad_out) 
  grad_b = np.sum(grad_out, axis=0)  
  return grad_x, grad_w, grad_b


def relu_forward(x):
  #computes forward pass for the ReLU nonlinearity
  out = np.maximum(0, x)
  cache = x
  return out, cache


def relu_backward(grad_out, cache):
  #computes backward pass for ReLU nonlinearity
  x = cache
  grad_x = grad_out * (x > 0)  
  return grad_x


def l2_loss(x, y):
  #computes the L2 loss and its gradient
  N = x.shape[0]
  diff = x - y
  loss = 0.5 * np.sum(diff * diff) / N
  grad_x = diff / N
  return loss, grad_x


def softmax_loss(x, y):
  #computes  loss and gradient for softmax loss function.
  N = x.shape[0]
  shifted_logits = x - np.max(x, axis=1, keepdims=True)
    
  exp_logits = np.exp(shifted_logits)
  probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
  correct_log_probs = -np.log(probs[np.arange(N), y])
  loss = np.sum(correct_log_probs) / N
    
  grad_x = probs.copy()
  grad_x[np.arange(N), y] -= 1
  grad_x /= N
    
  return loss, grad_x

def l2_regularization(w, reg):
  #computes loss and gradient for L2 regularization of a weight matrix
  loss = (reg / 2) * np.sum(w * w)
  grad_w = reg * w
  return loss, grad_w
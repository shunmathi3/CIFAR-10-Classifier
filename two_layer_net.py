import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward

class TwoLayerNet(Classifier):
   
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512, weight_scale=1e-3):
        #initalizing a two layer net
        self.params = {}
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def parameters(self):
        #a dict of all learnable parameters of this mod
        params = self.params
        return params

    def forward(self, X):
        # forward pass to compute classification scores for the input data
        h1, fc1_cache = fc_forward(X, self.params['W1'], self.params['b1'])
        h2, relu_cache = relu_forward(h1)
        scores, fc2_cache = fc_forward(h2, self.params['W2'], self.params['b2'])
        cache = (fc1_cache, relu_cache, fc2_cache)
        return scores, cache

    def backward(self, grad_scores, cache):
        #backward pass to compute gradients for all learnable parameters of the model
        grads = {}
        fc1_cache, relu_cache, fc2_cache = cache
        dh2, grads['W2'], grads['b2'] = fc_backward(grad_scores, fc2_cache)
        dh1 = relu_backward(dh2, relu_cache)
        _, grads['W1'], grads['b1'] = fc_backward(dh1, fc1_cache)
        return grads

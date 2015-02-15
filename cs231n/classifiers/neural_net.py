import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0, dropout=False):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  f = lambda x: x.clip(min = 0) #ReLU
  h1 = X.dot(W1) + b1
  if dropout:
	  p = 0.5 
	  U1 = (np.random.rand(*h1.shape) < p) / p # first dropout mask. Notice /p!
	  h1 *= U1 # drop!
  relu = f(h1)
  scores = relu.dot(W2) + b2
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################
  scores = scores.transpose()
  normalized_scores = scores - np.amax(scores, axis = 0)
  correct_scores = normalized_scores[[y, np.arange(y.size)]]
  score_exp = np.exp(normalized_scores)
  sum_exp = np.sum(score_exp, axis = 0)
  divide_exps = np.exp(correct_scores) / sum_exp
  avg_unreg_loss = np.sum(-np.log(divide_exps)) / N
  reg1 = 0.5 * reg * np.sum(W1 * W1)
  reg2 = 0.5 * reg * np.sum(W2 * W2)
  loss = avg_unreg_loss + reg1 + reg2
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  grads = {}
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
  dW1 = reg * W1
  dW2 = reg * W2
  
  dscores = np.exp(normalized_scores) / sum_exp
  dscores[[y, np.arange(y.size)]] -= 1  
  dW2 += (1/float(N)) * relu.transpose().dot(dscores.transpose()) 

  dL_dr = dscores.transpose().dot(W2.transpose())
  dr_dh1 = np.zeros_like(h1)
  dr_dh1[h1 > 0] = 1
  
  dW1 += (1/float(N)) * X.transpose().dot(dL_dr * dr_dh1)
  
  db1 = (1/float(N)) * (dL_dr * dr_dh1).transpose().dot(np.ones(N,))
  grads['W2'] = dW2
  grads['W1'] = dW1
  grads['b1'] = db1
  grads['b2'] = (1/float(N)) * dscores.dot(np.ones(N,))
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads

